import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_recommendations(group_type, user_interest, data, result_count=5):
    user_input = user_interest + '#' + group_type

    user_df = pd.DataFrame({'combined': [user_input]})
    combined_df = pd.concat([data['combined'], user_df['combined']], ignore_index=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df)

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:result_count + 1]

    recommended_indices = [i[0] for i in sim_scores]

    recommendations = data.iloc[recommended_indices].copy()
    recommendations['no'] = range(1, len(recommendations) + 1)
    return recommendations[['no', 'nama', 'deskripsi', 'alamat']]

if __name__ == "__main__":
    # input
    group_type = "teman"
    user_interest = "Tempat unik untuk berpetualang"
    result_count = 5

    data = pd.read_csv('yogyakarta.csv', delimiter=',')

    data['combined'] = data['deskripsi'] + '#' + data['kategori'] + '#' + data['group']

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)

    recommendations = get_recommendations(group_type, user_interest, data, result_count)

    print("\nRekomendasi tempat wisata:")
    print(recommendations)
