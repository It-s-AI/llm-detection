import argparse
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

from cc_net import jsonql, process_wet_file, split_by_lang, perplexity, minify

FILE_DIR = Path(__file__).parent
CUTOFF_CSV = FILE_DIR / "data" / "cutoff.csv"

class StreamMinifier(minify.Minifier):
    def __init__(self, remove: Optional[List[str]] = None, keep: Optional[List[str]] = None):
        super().__init__()
        self.remove = remove or []
        self.keep = keep or []

    def do(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        # Remove fields that are not needed
        for f in self.remove:
            doc.pop(f, None)
        
        # Keep only the specified fields
        if self.keep:
            doc = {k: v for k, v in doc.items() if k in self.keep}
        
        return doc

def stream_cc_segment(
    segment_url: str,
    output_dir: Path,
    lang_model: Path,
    lm_dir: Path,
    lang_whitelist: Optional[List[str]] = None,
    lang_threshold: float = 0.5,
    min_len: int = 300,
):
    # Set up the pipeline steps
    steps = []

    # Language identification
    steps.append(split_by_lang.Classifier(
        model=lang_model,
        field="raw_content",
        out_field="language",
        top=1,
        threshold=lang_threshold,
    ))

    # Language filtering
    if lang_whitelist:
        steps.append(jsonql.where(
            [lambda doc: doc.get("language") in set(lang_whitelist)]
        ))

    # SentencePiece tokenization
    steps.append(perplexity.MultiSentencePiece(
        {l: lm_dir / f"{l}.sp.model" for l in (lang_whitelist or ["en", "fr", "de"])},
        field="raw_content",
        output_field="tokenized",
        normalize=True,
    ))

    # Language model scoring
    steps.append(perplexity.DocLM(
        {l: lm_dir / f"{l}.arpa.bin" for l in (lang_whitelist or ["en", "fr", "de"])},
        field="tokenized",
        output_field="perplexity",
        normalize=False,
    ))

    # Perplexity bucketing
    steps.append(perplexity.PerplexityBucket(CUTOFF_CSV))

    # Minification (remove unnecessary fields)
    steps.append(StreamMinifier(remove=["tokenized"], keep=["url", "raw_content", "language", "perplexity", "bucket"]))

    # Set up the CC segment reader
    cc_reader = process_wet_file.CCSegmentsReader(
        [segment_url],
        min_len=min_len,
    )

    # Set up the output
    output_pattern = str(output_dir / "{language}_{bucket}.json.gz")
    steps.append(jsonql.split(pattern=output_pattern, mkdir=True))

    # Run the pipeline
    jsonql.run_pipes(
        *steps,
        inputs=cc_reader,
        processes=1,  # Increase this if you want to use multiple processes
        chunksize=100,
    )

def main():
    parser = argparse.ArgumentParser(description="Stream and process a CC segment")
    parser.add_argument("segment_url", type=str, help="URL of the CC segment to process")
    parser.add_argument("output_dir", type=Path, help="Directory to save processed files")
    parser.add_argument("--lang_model", type=Path, default=Path("bin/lid.bin"), help="Path to language identification model")
    parser.add_argument("--lm_dir", type=Path, required=True, help="Directory containing language models")
    parser.add_argument("--lang_whitelist", type=str, nargs="+", help="List of languages to process")
    parser.add_argument("--lang_threshold", type=float, default=0.5, help="Language identification threshold")
    parser.add_argument("--min_len", type=int, default=300, help="Minimum document length")

    args = parser.parse_args()

    stream_cc_segment(
        args.segment_url,
        args.output_dir,
        args.lang_model,
        args.lm_dir,
        args.lang_whitelist,
        args.lang_threshold,
        args.min_len,
    )

if __name__ == "__main__":
    main()