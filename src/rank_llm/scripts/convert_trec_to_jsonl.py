import argparse
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from pyserini.index.lucene import LuceneIndexReader
from pyserini.search import get_topics_with_reader
from tqdm import tqdm

from rank_llm.data import Candidate, DataWriter, Query, Request


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TREC run file to a retrieve_results.jsonl file."
    )
    parser.add_argument(
        "--trec_run_file", type=str, required=True, help="Path to the TREC run file."
    )
    parser.add_argument(
        "--topics_file", type=str, required=True, help="Path to the topics file."
    )
    parser.add_argument(
        "--index_path", type=str, required=True, help="Path to the Lucene index."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output retrieve_results.jsonl file.",
    )
    args = parser.parse_args()

    print(f"Loading index from {args.index_path}...")
    index_reader = LuceneIndexReader(args.index_path)

    print(f"Loading topics from {args.topics_file}...")
    topics_dict = get_topics_with_reader(
        "io.anserini.search.topicreader.TsvStringTopicReader", args.topics_file
    )
    topics = {str(k): v["title"] for k, v in topics_dict.items()}

    print(f"Reading TREC run file from {args.trec_run_file}...")
    requests_map = defaultdict(list)
    with open(args.trec_run_file, "r") as f:
        for line in tqdm(f):
            parts = line.strip().split()
            qid, _, docid, _, score, _ = parts

            try:
                document = index_reader.doc(docid)
                if document is None:
                    print(f"Warning: docid {docid} not found in index. Skipping.")
                    continue

                content_json = json.loads(document.raw())

                # Robust content extraction logic
                if "title" in content_json and "text" in content_json:
                    content = (
                        content_json["title"].strip() + ". " + content_json["text"]
                    )
                elif "title" in content_json and "segment" in content_json:
                    content = (
                        content_json["title"].strip() + ". " + content_json["segment"]
                    )
                elif "contents" in content_json:
                    content = content_json["contents"]
                elif "passage" in content_json:
                    content = content_json["passage"]
                else:
                    # Fallback to raw content if no known field is found
                    content = document.raw()
                    print(
                        f"Warning: No specific content field found for docid {docid}. Using raw content."
                    )

                candidate = Candidate(docid=docid, score=float(score), doc=content)
                requests_map[qid].append(candidate)
            except Exception as e:
                print(f"Error processing docid {docid} for qid {qid}: {e}")
                continue

    print("Constructing requests...")
    requests = []
    for qid, candidates in tqdm(requests_map.items()):
        if qid in topics:
            query_text = topics[qid]
            # Sort candidates by score descending, as is typical
            sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            request = Request(
                query=Query(text=query_text, qid=qid), candidates=sorted_candidates
            )
            requests.append(request)
        else:
            print(
                f"Warning: qid {qid} from run file not found in topics file. Skipping."
            )

    print(f"Writing {len(requests)} requests to {args.output_file}...")
    writer = DataWriter(requests)
    writer.write_in_jsonl_format(args.output_file)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
