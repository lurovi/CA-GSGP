import numpy as np


class TournamentSelection:

    @staticmethod
    def binary_tournament(pop, P, algorithm, **kwargs):
        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape

        if n_competitors != 2:
            raise Exception("Only pressure=2 allowed for binary tournament!")

        # the result this function returns
        S = np.full(n_tournaments, -1, dtype=np.int)

        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            # if the first individual is better, choose it
            if pop[a].F < pop[b].F:
                S[i] = a

            # otherwise take the other individual
            else:
                S[i] = b

        return S

    @staticmethod
    def generic_tournament(pop, P, algorithm, **kwargs):
        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape

        # the result this function returns
        S = np.full(n_tournaments, -1, dtype=np.int)

        # now do all the tournaments
        for i in range(n_tournaments):
            a = list(P[i])
            best_ind = -1
            best_fitness = 1e+12
            for j in range(len(a)):
                ind = a[j]
                if pop[ind].F[0] < best_fitness:
                    best_fitness = pop[ind].F[0]
                    best_ind = ind

            S[i] = best_ind

        return S

    @staticmethod
    def generic_tournament_best_2(pop, P, algorithm, **kwargs):
        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape

        # the result this function returns
        S = np.full(n_tournaments * 2, -1, dtype=np.int)

        t = 0

        # now do all the tournaments
        for i in range(n_tournaments):
            a = list(P[i])
            b = [(a[j], pop[a[j]].F[0]) for j in range(len(a))]
            b.sort(key=lambda x: x[1], reverse=False)
            S[t], S[t+1] = b[0][0], b[1][0]
            t += 2

        return S
