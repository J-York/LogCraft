import os
import numpy as np
import pandas as pd
import json
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def clean_log_templates(templates_list):
    cleaned_templates = []
    for template in templates_list:
        template = template.lower()
        template = re.sub(r'[<>*]', '', template)
        template = re.sub(r'<[^>]*>', '', template)
        template = re.sub(r'\\b\\d+\\b', 'num', template)
        template = re.sub(r'\\b(?:0x)?[a-fA-F\\d]+\\b', 'hexnum', template)
        template = re.sub(r'\\s+', ' ', template).strip()
        cleaned_templates.append(template)
    return cleaned_templates

def find_clusters(embeddings, counts, templates, threshold=0.9):
    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Initialize clusters using Union-Find data structure
    parent = list(range(len(embeddings)))
    
    # Function to find the root of the cluster
    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]
    
    # Function to union two clusters
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if counts[root_i] >= counts[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j
    
    # Merge clusters based on similarity threshold
    for i in tqdm(range(len(embeddings)), desc="Merging clusters"):
        for j in range(i+1, len(embeddings)):
            if sim_matrix[i][j] > threshold:
                union(i, j)
    
    # Find unique clusters
    unique_clusters = {}
    for i in range(len(embeddings)):
        root = find(i)
        if root not in unique_clusters:
            unique_clusters[root] = {'representative_template': templates[root], 'templates': set()}
        # unique_clusters[root]['indices'].add(i)
        unique_clusters[root]['templates'].add(templates[i])
    
    # Convert sets to lists for JSON serialization
    for cluster in unique_clusters.values():
        # cluster['indices'] = list(cluster['indices'])
        cluster['templates'] = list(cluster['templates'])
    
    return unique_clusters


def cluster(data_name):
    data_path = '../data_preprocessed'
    templates_file_path = os.path.join(data_path, data_name, f'{data_name}.log_templates.csv')
    embedding_model = SentenceTransformer('../models/stransformer/all-MiniLM-L6-v2')
    dataframe = pd.read_csv(templates_file_path)
    event_mapping = dataframe.set_index('EventTemplate')['EventId'].to_dict()
    templates_list = dataframe['EventTemplate'].to_list()
    templates_list_cleaned = clean_log_templates(templates_list)
    occurrences_list = dataframe['Occurrences'].to_list()
    embedding_list = embedding_model.encode(templates_list_cleaned)
    
    templates_dict = {template: {'Embedding': embedding, 'Occurrences': occurrences}
                      for template, embedding, occurrences in zip(templates_list, embedding_list, occurrences_list)}
    
    embeddings = np.array([value['Embedding'] for value in templates_dict.values()])
    counts = [value['Occurrences'] for value in templates_dict.values()]
    templates = list(templates_dict.keys())
    
    unique_clusters = find_clusters(embeddings, counts, templates, threshold=0.95)

    vocab_dict = {}
    id_vocab_dict = {}
    vocab_dict_unmerged = {}

    start_count = 2
    for cluster in unique_clusters.values():     
        for template in cluster["templates"]:
            vocab_dict[template] = start_count
        start_count += 1

    start_count = 2
    for cluster in unique_clusters.values():     
        for template in cluster["templates"]:
            id_vocab_dict[event_mapping[template]] = start_count
        start_count += 1

    start_count = 2
    for cluster in unique_clusters.values():
        for template in cluster["templates"]:
            vocab_dict_unmerged[template] = start_count
            start_count += 1

    value_set = set()
    # for key, value in reversed_dict.items():
    #     reversed_dict[key] = len(value_set) + 2
    #     value_set.add(value)
    
    # Save clusters and vocab to JSON files
    with open(os.path.join(data_path, data_name, 'vocab.json'), 'w+') as f:
        json.dump(vocab_dict, f, indent=2)
    
    with open(os.path.join(data_path, data_name, 'templates_clus.json'), 'w+') as f:
        json.dump(unique_clusters, f, indent=2)
    
    with open(os.path.join(data_path, data_name, 'vocab_nc.json'), 'w+') as f:
        json.dump(vocab_dict_unmerged, f, indent=2)

    with open(os.path.join(data_path, data_name, 'id_vocab.json'), 'w+') as f:
        json.dump(id_vocab_dict, f, indent=2)


if __name__ == '__main__':
    data_name = 'Thunderbird'
    cluster(data_name)