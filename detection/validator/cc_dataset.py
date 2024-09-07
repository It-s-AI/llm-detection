import argparse
import random
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
import glob
import gzip
import json
import requests

from cc_net import process_wet_file, jsonql, split_by_lang, perplexity, minify
from cc_net.stream_cc import StreamMinifier, CUTOFF_CSV  # Import StreamMinifier and CUTOFF_CSV from stream_cc
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


def get_2023_dumps() -> List[str]:
    url = "https://index.commoncrawl.org/collinfo.json"
    response = requests.get(url)
    response.raise_for_status()
    all_dumps = response.json()

    dumps_2023 = []
    for dump in all_dumps:
        dump = dump['id']
        if not dump.startswith('CC-MAIN-'):
            continue

        dump = dump[len('CC-MAIN-'):]
        if 2013 <= int(dump[:4]) <= 2022:
            dumps_2023.append(dump)
    return dumps_2023


class CCDataset:
    def __init__(
            self,
            dumps: List[str],
            num_segments: int,
            lang_model: Path,
            lm_dir: Path,
            lang_whitelist: Optional[List[str]] = None,
            lang_threshold: float = 0.5,
            min_len: int = 300,
            cache_dir: Optional[Path] = None,
            tmp_dir: Path = None,
    ):
        self.dumps = dumps
        self.num_segments = num_segments
        self.lang_model = lang_model
        self.lm_dir = lm_dir
        self.lang_whitelist = lang_whitelist
        self.lang_threshold = lang_threshold
        self.min_len = min_len
        self.cache_dir = cache_dir
        self.segments = self._select_random_segments()
        self.current_segment_index = 0
        self.current_segment_iter = None
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def _select_random_segments(self) -> List[str]:
        all_segments = []
        for dump in self.dumps:
            all_segments.extend(process_wet_file.cc_segments(dump, self.cache_dir))
        return random.sample(all_segments, min(self.num_segments, len(all_segments)))

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.current_segment_iter is None or not self.current_segment_iter:
            if self.current_segment_index >= len(self.segments):
                raise StopIteration

            segment = self.segments[self.current_segment_index]
            self.current_segment_index += 1
            self.current_segment_iter = self._process_segment(segment)

        try:
            return next(self.current_segment_iter)
        except StopIteration:
            self.current_segment_iter = None
            return self.__next__()

    def _process_segment(self, segment: str):
        logger.info(f"Processing segment: {segment}")
        steps = []

        # Language identification
        steps.append(split_by_lang.Classifier(
            model=self.lang_model,
            field="raw_content",
            out_field="language",
            top=1,
            threshold=self.lang_threshold,
        ))

        # Language filtering
        if self.lang_whitelist:
            steps.append(jsonql.where(
                [lambda doc: doc.get("language") in set(self.lang_whitelist)]
            ))

        # SentencePiece tokenization
        steps.append(perplexity.MultiSentencePiece(
            {l: self.lm_dir / f"{l}.sp.model" for l in (self.lang_whitelist or ["en", "fr", "de"])},
            field="raw_content",
            output_field="tokenized",
            normalize=True,
        ))

        # Language model scoring
        steps.append(perplexity.DocLM(
            {l: self.lm_dir / f"{l}.arpa.bin" for l in (self.lang_whitelist or ["en", "fr", "de"])},
            field="tokenized",
            output_field="perplexity",
            normalize=False,
        ))

        # Perplexity bucketing
        steps.append(perplexity.PerplexityBucket(CUTOFF_CSV))

        # Minification (remove unnecessary fields)
        steps.append(StreamMinifier(remove=["tokenized"], keep=["url", "raw_content", "language", "perplexity", "bucket"]))

        logger.info("Setting up CC segment reader")
        cc_reader = process_wet_file.CCSegmentsReader(
            [segment],
            min_len=self.min_len,
            cache_dir=self.cache_dir,
        )

        try:
            logger.info("Running pipeline")
            # Use the specified tmp_dir to store the output files
            output_pattern = str(self.tmp_dir / f"{{language}}_{{bucket}}.json.gz")
            steps.append(jsonql.split(pattern=output_pattern, mkdir=True))

            # Run the pipeline and save the output to files
            jsonql.run_pipes(*steps, inputs=cc_reader, processes=1, chunksize=100)

            # Create an iterator that alternates between head and middle buckets
            def alternating_bucket_iterator():
                bucket_files = defaultdict(list)
                for bucket in self.allowed_buckets:
                    bucket_files[bucket] = list(self.tmp_dir.glob(f"*_{bucket}.json.gz"))

                bucket_data = {bucket: [] for bucket in self.allowed_buckets}
                current_bucket_index = 0

                while True:
                    current_bucket = self.allowed_buckets[current_bucket_index]

                    # If the current bucket is empty, try to refill it
                    if not bucket_data[current_bucket]:
                        if not bucket_files[current_bucket]:
                            # If no more files for this bucket, move to the next bucket
                            current_bucket_index = (current_bucket_index + 1) % len(self.allowed_buckets)
                            continue

                        # Get a random file for the current bucket
                        file = random.choice(bucket_files[current_bucket])
                        bucket_files[current_bucket].remove(file)

                        # Read all lines from the file and shuffle them
                        with gzip.open(file, 'rt') as f:
                            bucket_data[current_bucket] = [json.loads(line) for line in f]
                        random.shuffle(bucket_data[current_bucket])

                    # If we still don't have data for this bucket, move to the next one
                    if not bucket_data[current_bucket]:
                        current_bucket_index = (current_bucket_index + 1) % len(self.allowed_buckets)
                        continue

                    # Yield a sample from the current bucket
                    yield bucket_data[current_bucket].pop()

                    # Move to the next bucket
                    current_bucket_index = (current_bucket_index + 1) % len(self.allowed_buckets)

            return alternating_bucket_iterator()

        except Exception as e:
            logger.error(f"Error processing segment {segment}: {traceback.format_exc()}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Stream and process random CC segments from 2023 dumps")
    parser.add_argument("--num_segments", type=int, default=10, help="Number of segments to process")
    parser.add_argument("--lang_model", type=Path, default=Path("bin/lid.bin"), help="Path to language identification model")
    parser.add_argument("--lm_dir", type=Path, required=True, help="Directory containing language models")
    parser.add_argument("--lang_whitelist", type=str, nargs="+", help="List of languages to process")
    parser.add_argument("--lang_threshold", type=float, default=0.5, help="Language identification threshold")
    parser.add_argument("--min_len", type=int, default=300, help="Minimum document length")
    parser.add_argument("--cache_dir", type=Path, help="Directory to cache downloaded segments")
    parser.add_argument("--tmp_dir", type=Path, default=Path("tmp_cc_dataset"), help="Temporary directory for processed files")

    args = parser.parse_args()

    # Get all dumps from 2023
    dumps_2023 = get_2023_dumps()
    logging.info(f"Found {len(dumps_2023)} dumps from 2023: {dumps_2023}")

    dataset = CCDataset(
        dumps=dumps_2023,
        num_segments=args.num_segments,
        lang_model=args.lang_model,
        lm_dir=args.lm_dir,
        lang_whitelist=args.lang_whitelist,
        lang_threshold=args.lang_threshold,
        min_len=args.min_len,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
    )

    for i, doc in enumerate(dataset):
        print(f"Document {i + 1}:")
        print(f"URL: {doc['url']}")
        print(f"Language: {doc['language']}")
        print(f"Perplexity: {doc['perplexity']}")
        print(f"Bucket: {doc['bucket']}")
        print(f"Content preview: {doc['raw_content'][:100]}...")
        print("\n" + "=" * 80 + "\n")
        time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()