import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class BookData:
    def __init__(self, books_file, users_file, ratings_file):
        self.books = pd.read_csv(books_file)
        self.users = pd.read_csv(users_file)
        self.ratings = pd.read_csv(ratings_file)

    def _clean_books(self):
        required_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']
        self.books = self.books[required_columns]
        self.books.rename(columns={
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
            "Image-URL-L": "img_url"
        }, inplace=True)

    def _clean_ratings(self):
        self.ratings.rename(columns={
            "User-ID": "user_id",
            "Book-Rating": "rating"
        }, inplace=True)
        user_counts = self.ratings['user_id'].value_counts()
        frequent_users = user_counts[user_counts > 50].index
        self.ratings = self.ratings[self.ratings['user_id'].isin(frequent_users)]
    
    def _categorize_ratings(self):
        conditions = [
            (self.ratings['rating'] >= 0) & (self.ratings['rating'] <= 3),
            (self.ratings['rating'] > 3) & (self.ratings['rating'] <= 7),
            (self.ratings['rating'] > 7) & (self.ratings['rating'] <= 10)
        ]
        choices = ['low', 'medium', 'high']
        self.ratings['rating_class'] = np.select(conditions, choices, default='medium')
    
    def preprocess_data(self):
        self._clean_books()
        self._clean_ratings()
        self._categorize_ratings() 

class BookModel:
    def __init__(self, model_type='knn'):
        self.model_type = model_type
        if model_type == 'knn':
            self.model = NearestNeighbors(algorithm='brute')
        elif model_type == 'knn_classifier':
            self.model = KNeighborsClassifier()
        elif model_type == 'svd':
            self.model = TruncatedSVD(n_components=50)
            self.book_embeddings = None  # Placeholder for SVD transformed data

    def train_model(self, book_sparse, labels=None):
        if self.model_type in ['knn', 'svd']:
            if isinstance(self.model, TruncatedSVD):
                self.book_embeddings = self.model.fit_transform(book_sparse)
            else:
                self.model.fit(book_sparse)
        elif self.model_type == 'knn_classifier':
            if labels is None:
                raise ValueError("Labels required for training KNN Classifier.")
            self.model.fit(book_sparse, labels)
            

class RecommendationSingleton:
    _instance = None

    def __init__(self, books_file='Books.csv', users_file='Users.csv', ratings_file='Ratings.csv', model_type='knn'):
        if RecommendationSingleton._instance is None:
            RecommendationSingleton._instance = self
            self._initialized = False
        else:
            return

        if not self._initialized:
            self.setup(books_file, users_file, ratings_file, model_type)

    def recommend_books(self, book_name, n_recommendations=5):
        try:
            book_id = np.where(self.books_data.book_pivot.index == book_name)[0][0]
            if self.model.model_type == 'svd':
                # Use cosine similarity on SVD embeddings
                cosine_sim = cosine_similarity(self.model.book_embeddings)
                sim_scores = list(enumerate(cosine_sim[book_id]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:n_recommendations + 1]  # Exclude self
                book_indices = [i[0] for i in sim_scores]
            else:
                # Use KNN model's kneighbors method
                _, suggestion = self.model.model.kneighbors(self.books_data.book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
                book_indices = suggestion[0][1:]  # Exclude self

            suggestions = [self.books_data.book_pivot.index[i] for i in book_indices]
            return suggestions
        except Exception as e:
            print(f"An error occurred during book recommendation: {e}")
            return None
    
    def setup(self, books_file, users_file, ratings_file, model_type):
        try:
            self.books_data = BookData(books_file, users_file, ratings_file)
            self.books_data.preprocess_data()

            # Merging ratings with books to include 'title' in the ratings
            ratings_with_books = self.books_data.ratings.merge(self.books_data.books[['ISBN', 'title']], on="ISBN")
            num_rating = ratings_with_books.groupby('title')['rating'].count().reset_index()
            num_rating.rename(columns={'rating': 'num_of_ratings'}, inplace=True)
            final_rating = ratings_with_books.merge(num_rating, on='title')
            final_rating = final_rating[final_rating['num_of_ratings'] >= 70]
            final_rating.drop_duplicates(['user_id', 'title'], inplace=True)


            # Create pivot table with 'title' as index
            self.books_data.book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating').fillna(0)
            self.books_data.book_sparse = csr_matrix(self.books_data.book_pivot.values)

            self.model = BookModel(model_type)
            # Grouping the final_rating by 'title' to ensure one unique label per title
            # Assuming that 'rating_class' is added, you would first need to define how you want to handle multiple classes for a single title.
            # For example, by taking the most frequent class:
            labels = final_rating.groupby('title').agg({'rating_class': lambda x: x.mode()[0]}).squeeze()
            labels = labels.reindex(self.books_data.book_pivot.index).fillna('medium')  # Filling missing values if any

            if model_type == 'knn_classifier':
                # Use 'final_rating' directly if it's already been deduplicated and aggregated
                labels = final_rating.groupby('title').agg({'rating_class': lambda x: x.mode()[0]}).squeeze()

                # Reindex labels to match the book pivot index, handle any potential issues with duplicates here
                try:
                    labels = labels.reindex(self.books_data.book_pivot.index).fillna('medium')  # Filling missing values if any

                except Exception as e:
                    print(f"Reindexing error: {e}")

            self.model.train_model(self.books_data.book_sparse, labels)


        except Exception as e:
            print(f"An error occurred during singleton initialization: {e}")

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls(*args, **kwargs)
        return cls._instance


class UserInteraction:
    def __init__(self):
        self.recommendation_system = None
        self.model_type = None

    def setup(self):
        print("Welcome to the Book Recommendation System Setup")
        books_file = input("Enter the path to the books dataset (default 'Books.csv'): ") or 'Books.csv'
        users_file = input("Enter the path to the users dataset (default 'Users.csv'): ") or 'Users.csv'
        ratings_file = input("Enter the path to the ratings dataset (default 'Ratings.csv'): ") or 'Ratings.csv'
        
        print("Choose a model type: \n1. KNN (default)\n2. KNN Classifier\n3. SVD")
        model_choice = input("Enter your choice (1, 2, or 3): ")
        self.model_type = 'knn'
        if model_choice == '2':
            self.model_type = 'knn_classifier'
        elif model_choice == '3':
            self.model_type = 'svd'
        
        self.recommendation_system = RecommendationSingleton.get_instance(books_file, users_file, ratings_file, self.model_type)

    def start(self):
        self.setup()
        while True:
            book_name = input("Enter the name of a book to get recommendations (or 'exit' to quit): ")
            if book_name.lower() == 'exit':
                print("Exiting...")
                break
            recommendations = self.recommendation_system.recommend_books(book_name)
            if recommendations:
                print(f"You searched for '{book_name}'\n")
                print("The suggested books are: \n")
                for i, book in enumerate(recommendations):
                    print(f"{i + 1}. {book}")
                
                # Plotting and calculating accuracy
                try:
                    if self.model_type == 'knn' or self.model_type == 'knn_classifier' or self.model_type == 'svd':
                        # Plotting for KNN, KNN Classifier, or SVD model
                        plt.figure(figsize=(10, 6))
                        plt.barh(np.arange(len(recommendations)), np.arange(len(recommendations)) + 1, color='skyblue')
                        plt.xlabel('Recommendation Rank')
                        plt.ylabel('Book Title')
                        plt.title('Recommendation Rank for Books')
                        plt.yticks(np.arange(len(recommendations)), [book[:50] + '...' if len(book) > 50 else book for book in recommendations])
                        plt.gca().invert_yaxis()  # Invert y-axis to have the top recommendation at the top
                        plt.tight_layout()
                        plt.show()

                        # Calculate accuracy (not applicable for these models)
                        accuracy = None

                except Exception as e:
                    print(f"An error occurred during plotting or accuracy calculation: {e}")
                    
            else:
                print("Book not found. Please try another one.")

def main():
    user_interaction = UserInteraction()
    user_interaction.start()

if __name__ == "__main__":
    main()
