from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import re

def cosine_similarity_vectors(vectors):
    return cosine_similarity(vectors)

def clean_log_templates(templates_list):
    cleaned_templates = []
    for template in templates_list:
        # Convert to lowercase
        template = template.lower()
        # Remove <, >, and *
        template = re.sub(r'[<>*]', '', template)
        # Replace all numbers with 'num'
        template = re.sub(r'\b\d+\b', 'num', template)
        # Replace all hexadecimal numbers with 'hexnum'
        template = re.sub(r'\b(?:0x)?[a-fA-F\d]+\b', 'hexnum', template)
        # Remove extra spaces
        template = re.sub(r'\s+', ' ', template).strip()
        cleaned_templates.append(template)
    return cleaned_templates

def find_clusters(templates_dict, threshold=0.8):
    # Extract embeddings and counts from the dictionary
    embeddings = np.array([value['Embedding'] for value in templates_dict.values()])
    counts = [value['Occurrences'] for value in templates_dict.values()]
    templates = list(templates_dict.keys())
    
    # Calculate cosine similarity matrix
    sim_matrix = cosine_similarity_vectors(embeddings)
    
    # Initialize clusters
    clusters = {}
    for i in range(len(embeddings)):
        clusters[i] = {'indices': set([i]), 'max_count': counts[i], 'representative': i}
    
    # Merge clusters based on similarity threshold
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if sim_matrix[i][j] > threshold:
                # Merge clusters if similarity is above threshold
                clusters[i]['indices'].update(clusters[j]['indices'])
                # Update representative if count is higher
                if counts[j] > clusters[i]['max_count']:
                    clusters[i]['max_count'] = counts[j]
                    clusters[i]['representative'] = j
                # Point all merged indices to the representative cluster
                for index in clusters[i]['indices']:
                    clusters[index] = clusters[i]
    
    # Deduplicate clusters
    unique_clusters = {}
    for cluster in clusters.values():
        rep = cluster['representative']
        unique_clusters[rep] = unique_clusters.get(rep, {'indices': set(), 'representative_template': None})
        unique_clusters[rep]['indices'].update(cluster['indices'])
        unique_clusters[rep]['representative_template'] = templates[rep]
    
    # Convert sets to lists for output
    for cluster_id, cluster_info in unique_clusters.items():
        cluster_info['indices'] = list(cluster_info['indices'])
    
    # Create a dictionary to hold the final clusters with templates
    final_clusters = {}
    for cluster_id, cluster_info in unique_clusters.items():
        final_clusters[cluster_id] = [templates[i] for i in cluster_info['indices']]
    
    return final_clusters


if __name__ == '__main__':
    templates_file_path = '../data_preprocessed/BGL_ncom/BGL.log_templates.csv'
    embedding_model = SentenceTransformer('../models/stransformer/all-MiniLM-L6-v2')
    dataframe = pd.read_csv(templates_file_path)
    templates_list = dataframe['EventTemplate'].to_list()
    templates_list_cleaned = clean_log_templates(templates_list)
    print('templates_before counts: ', len(templates_list))
    # print(templates_list)
    occurrences_list = dataframe['Occurrences'].to_list()
    embedding_list = embedding_model.encode(templates_list)
    templates_dict = {}
    for i in range(len(templates_list)):
        templates_dict[templates_list[i]] = {}
        templates_dict[templates_list[i]]['Embedding'] = embedding_list[i]
        templates_dict[templates_list[i]]['Occurrences'] = occurrences_list[i]

    # Find clusters with a threshold of 0.9
    clusters = find_clusters(templates_dict, threshold=1)

    print('templates_after counts: ', len(clusters))

    reversed_dict = {value: int(key+2) for key, values in clusters.items() for value in values}


    with open('../data_preprocessed/BGL_ncom/templates_clus.json', 'w+') as f:
        json.dump(clusters, f, indent=2)

    with open('../data_preprocessed/BGL_ncom/vocab.json', 'w+') as f:
        json.dump(reversed_dict, f, indent=2)
    