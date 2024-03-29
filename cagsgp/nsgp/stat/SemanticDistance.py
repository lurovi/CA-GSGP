import statistics
import numpy as np


class SemanticDistance:
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def one_matrix_zero_diagonal(N: int) -> list[list[float]]:
        return [ [0.0 if i == j else 1.0 for j in range(N)] for i in range(N)]
    
    @staticmethod
    def one_matrix(N: int) -> list[list[float]]:
        return [ [1.0 for j in range(N)] for i in range(N)]
    
    @staticmethod
    def zero_matrix(N: int) -> list[list[float]]:
        return [ [0.0 for j in range(N)] for i in range(N)]

    @staticmethod
    def sum_of_all_elem_in_matrix(v: list[list[float]]) -> float:
        return float(sum([sum(l) for l in v]))

    @staticmethod
    def sum_of_all_elem_in_numpy_matrix(v: np.ndarray) -> float:
        return float(np.sum(v))
    
    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))
    
    @staticmethod
    def self_dot_product(v1: np.ndarray) -> float:
        return float(np.dot(v1, v1))

    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.square(v1 - v2))))
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    @staticmethod
    def geometric_center(vectors: list[np.ndarray]) -> np.ndarray:
        semantic_matrix: np.ndarray = np.stack(vectors, axis=0)
        return np.mean(semantic_matrix, axis=0)

    @staticmethod
    def global_moran_I(vectors: list[np.ndarray], w: list[list[float]]) -> float:
        N: int = len(vectors)
        W: float = SemanticDistance.sum_of_all_elem_in_matrix(w)
        gc: np.ndarray = SemanticDistance.geometric_center(vectors)

        numerator: float = sum([ sum([w[i][j] * SemanticDistance.dot_product(vectors[i] - gc, vectors[j] - gc) for j in range(N)]) for i in range(N)])
        
        denominator: float = sum([SemanticDistance.self_dot_product(vectors[i] - gc) for i in range(N)])
        denominator = denominator if denominator != 0 else 1e-12

        return ( float(N) / W ) * (numerator / denominator)
    
    @staticmethod
    def global_moran_I_coef(vectors: list[np.ndarray], w: list[list[float]], coef: float) -> float:
        N: int = len(vectors)
        gc: np.ndarray = SemanticDistance.geometric_center(vectors)

        numerator: float = sum([ sum([w[i][j] * SemanticDistance.dot_product(vectors[i] - gc, vectors[j] - gc) for j in range(N)]) for i in range(N)])
        
        denominator: float = sum([SemanticDistance.self_dot_product(vectors[i] - gc) for i in range(N)])
        denominator = denominator if denominator != 0 else 1e-12

        return coef * (numerator / denominator)

    @staticmethod
    def compute_multi_euclidean_distance_from_list(vectors: list[np.ndarray]) -> float:
        semantic_matrix: np.ndarray = np.stack(vectors, axis=0)
        multi_population_semantic_distance: float = float(np.sqrt(np.sum(np.var(semantic_matrix, axis=0))))
        return multi_population_semantic_distance
    
    @staticmethod
    def compute_distances_between_vector_at_index_and_rest_of_the_list(idx: int, vectors: list[np.ndarray]) -> list[float]:
        if not 0 <= idx < len(vectors):
            raise IndexError(f'{idx} is out of range as index of the list of semantic vectors, which length is {len(vectors)}.')
        distances: list[float] = []

        for i in range(len(vectors)):
            if i != idx:
                distances.append(SemanticDistance.euclidean_distance(vectors[idx], vectors[i]))

        return distances
    
    @staticmethod
    def compute_distances_stats_among_vectors(vectors: list[np.ndarray]) -> dict[str, float]:
        mean_distances: list[float] = []

        distance_matrix: list[list[float]] = [[None for j in range(len(vectors))] for i in range(len(vectors))]

        for i in range(len(vectors)):
            i_distances: list[float] = []
            for j in range(len(vectors)):
                if i == j:
                    distance_matrix[i][j] = 0.0
                else:
                    if distance_matrix[i][j] is None:
                        temp: float = SemanticDistance.euclidean_distance(vectors[i], vectors[j])
                        distance_matrix[i][j] = temp
                        distance_matrix[j][i] = temp
                    i_distances.append(distance_matrix[i][j])
            mean_distances.append(statistics.mean(i_distances))

        return SemanticDistance.compute_stats(mean_distances)
    
    @staticmethod
    def compute_stats_all_distinct_distances(vectors: list[np.ndarray]) -> dict[str, float]:
        distances: list[float] = []

        for i in range(len(vectors) - 1):
            for j in range(i + 1, len(vectors)):
                distances.append(SemanticDistance.euclidean_distance(vectors[i], vectors[j]))

        return SemanticDistance.compute_stats(distances)

    @staticmethod
    def compute_stats(distances: list[float]) -> dict[str, float]:
        stats: dict[str, float] = {}

        stats['mean'] = statistics.mean(distances)
        stats['median'] = statistics.median(distances)
        stats['max'] = max(distances)
        stats['min'] = min(distances)
        stats['var'] = statistics.pvariance(distances, mu=stats['mean'])
        stats['q1'] = float(np.percentile(distances, 25))
        stats['q3'] = float(np.percentile(distances, 75))

        return stats
    
    @staticmethod
    def compute_stats_only_integer(values: list[int]) -> dict[str, int]:
        stats: dict[str, int] = {}
        sorted_values: list[int] = sorted(values, reverse=False)

        stats['mean'] = sum(values) // len(values)
        stats['median'] = sorted_values[len(values) // 2]
        stats['max'] = sorted_values[-1]
        stats['min'] = sorted_values[0]
        stats['var'] = 0
        stats['q1'] = sorted_values[int(len(values) * 0.25)]
        stats['q3'] = sorted_values[int(len(values) * 0.75)]

        return stats
    