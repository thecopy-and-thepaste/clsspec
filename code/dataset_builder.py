"""
In this script we gather species names from the GBIF endpoint: species.

We build the plain taxonomic tree from a set of kingdoms
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import pydash
import requests
import random
import time
import multiprocessing

from multiprocessing import Pool
from functools import partial

from pathlib import Path

from pdb import set_trace as bp

LIMIT = 500
SLEEP_TIME = 15
NUM_PROCESSES = multiprocessing.cpu_count() - 1


# region fetch

def fetch_dataset(tree: dict,
                  dest_file: Path):
    prev_trees = []
    from_prev_tree = False
    curr_tree = tree
    found_empty_branch = True
    next_keys = []

    while found_empty_branch:
        found_empty_branch = False

        if not from_prev_tree:
            next_keys = [k for k in curr_tree.keys()]

        from_prev_tree = False

        if len(next_keys) > 0:
            found_empty_branch = True
            key = next_keys.pop()

            prev_trees.append((curr_tree, next_keys))

            if "children" not in curr_tree[key]:
                if not curr_tree[key].get("hasDescendants", True):
                    curr_tree[key]["children"] = {}
                else:
                    children = fetch_children(key)
                    children = {ch["key"]: ch for ch in children}

                    curr_tree = curr_tree[key]
                    curr_tree["children"] = children
                    curr_tree = curr_tree["children"]

                with open(dest_file, mode="w") as _f:
                    json.dump(tree, _f)
            elif len(curr_tree[key]["children"]) > 0:
                curr_tree = curr_tree[key]["children"]
            else:
                tmp = prev_trees.pop()
                curr_tree = tmp[0]
                next_keys = tmp[1]
                from_prev_tree = True
        else:
            if len(prev_trees) > 0:
                found_empty_branch = True

                tmp = prev_trees.pop()
                curr_tree = tmp[0]
                next_keys = tmp[1]
                from_prev_tree = True


def fetch_children(species_id: int):
    offset = 0
    last_page = False
    results = []

    while not last_page:
        time.sleep(random.randint(0, SLEEP_TIME))

        query_url = f"https://api.gbif.org/v1/species/{species_id}/children?offset={offset}&limit={LIMIT}"
        r = requests.get(url=query_url)
        response = r.json()

        last_page = response["endOfRecords"]
        page_results = response["results"]

        print(f"Fetch {species_id}: offset {offset}")

        page_results = pydash.chain(page_results)\
            .map(lambda x: {
                "key": x["key"],
                "scientificName": x.get("scientificName", "NA"),
                "canonicalName": x.get("canonicalName", "NA"),
                "taxonID": x.get("taxonID", "NA"),
                "hasDescendants": int(x.get("numDescendants", 0)) > 0,
            })\
            .value()

        results += page_results
        offset += LIMIT

    return results
# endregion

# region plain dataset


def store_plain_dataset(tree_file: dict,
                        dest_file: Path):
    txt_file = plain_dataset(tree_file=tree_file)


    # pydash.chain(txt_file[0]).map(lambda x: ",".join(x)).value()
    with open(dest_file, mode="w") as _f:
        _f.write(txt_file)


def plain_dataset(tree_file: dict, parents:list = []):
    acc = ""
    keys = tree_file.keys()

    if len(keys) == 0:
        # if len(parents) > 2:
        #     bp()
        return ",".join(parents) + "\n"

    for key in keys:
        curr_tree = tree_file[key]
        name = curr_tree["canonicalName"]
        children = curr_tree.get("children", {})
        
        next_parent = parents + [name]
        
        prev = plain_dataset(children, next_parent)
        # prev = pydash.chain(prev).flatten().value()

        acc += prev

    return acc
# endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--start_tree",
                        help="Names to build the taxonomic tree")
    parser.add_argument("-d", "--dest_file",
                        help=(f"Destination folder to store the taxonomic tree."))
    parser.add_argument("-p", "--plain", action='store_true',
                        help=(f"Make txt file of json file"))

    ARGS = parser.parse_args()

    start_tree = ARGS.start_tree
    dest_file = ARGS.dest_file
    plain = ARGS.plain

    with open(start_tree) as _f:
        start_tree = json.load(_f)

    if plain:
        store_plain_dataset(start_tree,
                            Path(dest_file))
    else:
        fetch_dataset(start_tree,
                      Path(dest_file))
