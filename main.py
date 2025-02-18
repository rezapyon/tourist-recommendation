from sklearn.metrics.pairwise import cosine_similarity
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

def get_recommendations(group_type, user_interest, data, result_count=5):
    user_input = user_interest + '#' + group_type

    user_df = pandas.DataFrame({'combined': [user_input]})
    combined_df = pandas.concat([data['combined'], user_df['combined']], ignore_index=True)

    tf_idf_vector = TfidfVectorizer()
    tf_idf_mx = tf_idf_vector.fit_transform(combined_df)

    similar_cosine = cosine_similarity(tf_idf_mx[-1], tf_idf_mx[:-1])

    similar_scores = list(enumerate(similar_cosine[0]))
    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
    similar_scores = similar_scores[1:result_count + 1]

    recomm_index = [i[0] for i in similar_scores]

    recommendations = data.iloc[recomm_index].copy()
    recommendations['no'] = range(1, len(recommendations) + 1)
    return recommendations[['no', 'nama', 'deskripsi', 'alamat']]

if __name__ == "__main__":
    # input
    group_type = "teman"
    user_interest = "Tempat unik untuk berpetualang"
    result_count = 5

    place_data_frame = pandas.read_csv('yogyakarta.csv', delimiter=',')

    place_data_frame['combined'] = place_data_frame['deskripsi'] + '#' + place_data_frame['kategori'] + '#' + place_data_frame['group']

    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.expand_frame_repr', False)
    pandas.set_option('display.max_colwidth', None)

    recommendations = get_recommendations(group_type, user_interest, place_data_frame, result_count)

    print("\nRekomendasi tempat wisata:")
    print(recommendations)
