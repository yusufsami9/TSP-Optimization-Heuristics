# -*- coding: utf-8 -*-
"""
TSP + NN + Outlier Insertion + GRASP(Insertion) + 2-opt

Author: Yusuf Sami

"""

import math
import numpy as np
import random
import matplotlib.pyplot as plt

class Point2D:
    """Class for representing a point in 2D space"""
    def __init__(self, id_, x, y):
        self.id = id_
        self.x = x
        self.y = y


    def getDistance(c1, c2):
        dx = c1.x - c2.x
        dy = c1.y - c2.y
        return math.sqrt(dx ** 2 + dy ** 2)



class TSP:
    """
    Class for representing a Traveling Salesman Problem

    Attributes
    ----------
    nCities : int
        number of cities
    cities : list[int]
        list of city indices 0..n-1
    distMatrix : np.ndarray
        symmetric n x n matrix with pairwise distances
    """

    # ------------- constructor & reading -------------
    def __init__(self, tspFileName):
        """
        Reads a .tsp file and constructs an instance.
        We assume that it is an Euclidian TSP

        Parameters
        ----------
        tspFileName : str
            name of the file
        """
        points = list()  # add all points to list
        f = open(tspFileName)
        for line in f.readlines()[6:-1]:  # start reading from line 7, skip last line
            asList = line.split()
            floatList = list(map(float, asList))

            id = int(floatList[0]) - 1  # convert to int, subtract 1 because Python indices start from 0
            x = floatList[1]
            y = floatList[2]

            c = Point2D(id, x, y)
            points.append(c)
        f.close()

        print("Read in all points, start computing distance matrix")

        self.nCities = len(points)
        self.cities = list(range(self.nCities))
        self.points = points  # try

        # compute distance matrix, assume Euclidian TSP
        self.distMatrix = np.zeros((self.nCities, self.nCities))  # init as nxn matrix
        for i in range(self.nCities):
            for j in range(i + 1, self.nCities):
                distItoJ = Point2D.getDistance(points[i], points[j])
                self.distMatrix[i, j] = distItoJ
                self.distMatrix[j, i] = distItoJ

        print("Finished computing distance matrix")



    # ------------- basic helpers -------------
    def getCitiesCopy(self):
        return self.cities.copy()

    def isFeasible(self,tour):
        """
        Checks if tour is feasible

        Parameters
        ----------
        tour : list of integers
            order in which cities are visited. For a 4-city TSP, an example tour is [3, 1, 4, 2]

        Returns
        -------
        bool
            TRUE if feasible, FALSE if infeasible.

        """
        #first check if the length of the tour is correct
        if len(tour)!=self.nCities:
            print("Length of tour incorrect")
            return False
        else:
            #check if all cities in the tour
            for city in self.cities:
                if city not in tour:
                    return False
        return True

    def computeCosts(self, tour):
        """
        Computes the costs of a tour

        Parameters
        ----------
        tour : list of integers
            order of cities.

        Returns
        -------
        costs : int
            costs of tour.

        """
        costs = 0
        for i in range(len(tour) - 1):
            costs += self.distMatrix[tour[i], tour[i + 1]]

        # add the costs to complete the tour back to the start
        costs += self.distMatrix[tour[-1], tour[0]]
        return costs

    def tour_cost(self, tour):
        return self.computeCosts(tour)

    def evaluateSolution(self, tour):
        if self.isFeasible(tour):
            costs = self.computeCosts(tour)
            print("The solution is feasible with costs " + str(costs))
        else:
            print("The solution is infeasible")

    # ------------- Nearest Neighbour -------------
    def getTour_NN(self, start):
        """
        Performs the nearest neighbour algorithm

        Parameters
        ----------
        start : int
            starting point of the tour

        Returns
        -------
        tour : list of ints
            order in which the cities are visitied.

        """
        tour = [start]
        notInTour = self.cities.copy()
        notInTour.remove(start)


        for i in range(self.nCities - 1):
            curCity = tour[i]
            closestDist = -1  # initialize with -1
            closestCity = None  # initialize with None

            # find closest city not yet in tour
            for j in notInTour:
                dist = self.distMatrix[curCity][j]
                if dist < closestDist or closestCity is None:
                    # update the closest city and distance
                    closestDist = dist
                    closestCity = j

            tour.append(closestCity)
            notInTour.remove(closestCity)



        return tour
    # 2nd step in the assignment, use it while reporting
    def run_NN_all_starts(self):

        #Run NN from every start; return list of (start, cost) sorted by cost.

        results = []
        for s in self.cities:
            tour = self.getTour_NN(s)
            results.append((s, self.tour_cost(tour)))
        results.sort(key=lambda x: x[1])
        return results

    # ------------- Outlier Insertion (deterministic) -------------
    def delta_insertion_cost(self, tour, city, pos):
        """
        Cost increase when inserting 'city' between tour[pos] and tour[pos+1]
        (with wrap-around).
        """
        n = len(tour)
        a = tour[pos]
        b = tour[(pos + 1) % n]
        return (self.distMatrix[a, city] + self.distMatrix[city, b]
                - self.distMatrix[a, b])

    def getTour_OutlierInsertion(self, start):
        """
        Deterministic Outlier Insertion as described in the assignment:
        1) Start with [start] joined with farthest city from start.
        2) While unvisited exists:
           a) choose city k that maximizes nearest-to-tour distance: max_k min_{t in tour} d(k,t)
           b) insert k in position that minimizes insertion delta
        """
        notIn = set(self.cities)
        notIn.remove(start)

        # farthest from start
        far_city = max(notIn, key=lambda j: self.distMatrix[start, j])
        tour = [start, far_city]
        notIn.remove(far_city)

        while notIn:
            # pick city that is 'furthest to any city in the tour' (i.e., maximize nearest-to-tour distance)
            def nearest_to_tour_dist(k):
                return min(self.distMatrix[k, t] for t in tour)
            k_star = max(notIn, key=nearest_to_tour_dist)

            # insert where it increases the length the least
            best_pos = None
            best_inc = None
            for pos in range(len(tour)):
                inc = self.delta_insertion_cost(tour, k_star, pos)
                if (best_inc is None) or (inc < best_inc):
                    best_inc = inc
                    best_pos = pos
            tour.insert(best_pos + 1, k_star)
            notIn.remove(k_star)

        return tour

    # ------------- GRASPed Outlier Insertion -------------

    def _build_rcl_by_quality(self, scored_list, alpha, maximize=True):
        """
        Build RCL by relative quality band.
        scored_list: [(item, score), ...]
        alpha in [0,1]: 0 => only best; 1 => everyone.
        maximize=True: larger score better; False: smaller better.
        Returns list of items (no scores).
        """
        if not scored_list:  # safeguard
            return []
        # sort by score (best first if maximize=True)
        scored_list = sorted(scored_list, key=lambda x: x[1], reverse=maximize)
        best = scored_list[0][1]
        worst = scored_list[-1][1]
        if maximize:
            thr = best - alpha * (best - worst)
            rcl = [it for (it, s) in scored_list if s >= thr]
        else:
            thr = best + alpha * (worst - best)
            rcl = [it for (it, s) in scored_list if s <= thr]
        return rcl

    def _build_rcl_by_k(self, scored_list, k, maximize=True):
        """
        Build RCL by top-k (fallback or user preference).
        """
        if not scored_list:
            return []
        scored_list = sorted(scored_list, key=lambda x: x[1], reverse=maximize)
        k = max(1, int(k))
        k = min(k, len(scored_list))
        return [it for (it, _) in scored_list[:k]]

    def getTour_GRASPedInsertion(self,
                                 start,
                                 alpha_city=0.30,
                                 alpha_pos=0.00,
                                 seed=None,
                                 rcl_city_len=None,
                                 rcl_pos_len=None):
        """
        GRASP Outlier Insertion supporting both alpha-based (quality) RCL and top-k RCL.
        - If rcl_city_len is provided, use top-k for city selection; else use alpha_city.
        - If rcl_pos_len is provided, use top-k for position selection; else use alpha_pos.
        """
        import random
        if seed is not None:
            random.seed(seed)

        notIn = set(self.cities)
        notIn.remove(start)

        # start pair
        far_city = max(notIn, key=lambda j: self.distMatrix[start, j])
        tour = [start, far_city]
        notIn.remove(far_city)

        while notIn:
            # (i) city scores: score = min distance to any city in current tour (maximize)
            city_scores = [(k, min(self.distMatrix[k, t] for t in tour)) for k in notIn]
            if rcl_city_len is not None:
                rcl_cities = self._build_rcl_by_k(city_scores, rcl_city_len, maximize=True)
            else:
                rcl_cities = self._build_rcl_by_quality(city_scores, alpha_city, maximize=True)
            k_star = random.choice(rcl_cities)

            # (ii) position scores: score = delta increase (minimize)
            pos_scores = [(pos, self.delta_insertion_cost(tour, k_star, pos)) for pos in range(len(tour))]
            if rcl_pos_len is not None:
                rcl_pos = self._build_rcl_by_k(pos_scores, rcl_pos_len, maximize=False)
            else:
                rcl_pos = self._build_rcl_by_quality(pos_scores, alpha_pos, maximize=False)
            pos_star = random.choice(rcl_pos)

            tour.insert(pos_star + 1, k_star)
            notIn.remove(k_star)

        return tour

    # ------------- 2-opt -------------
    def _two_opt_gain(self, tour, i, j):
        """
        Replace edges (i,i+1) and (j,j+1) by (i,j) and (i+1,j+1); return delta (after - before).
        Negative delta ⇒ improvement.
        """
        n = len(tour)
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        before = self.distMatrix[a, b] + self.distMatrix[c, d]
        after = self.distMatrix[a, c] + self.distMatrix[b, d]
        return after - before

    def isTwoOpt(self, tour):
        """Return True iff no improving 2-opt move exists."""
        n = len(tour)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # adjacent edges (wrap) not allowed
                if self._two_opt_gain(tour, i, j) < -1e-12:
                    return False
        return True

    def makeTwoOpt(self, tour, first_improvement=False):
        """
        Apply 2-opt until no improving move remains.
        If first_improvement=True, stop at first found improvement per scan (faster).
        If False, do best-improvement per scan (slower but sometimes better).
        """
        n = len(tour)
        tour = tour[:]  # work on a copy
        improved = True
        while improved:
            improved = False
            best_gain = 0.0
            best_move = None
            for i in range(n):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    gain = self._two_opt_gain(tour, i, j)
                    if gain < -1e-12:
                        if first_improvement:
                            # reverse segment (i+1 .. j)
                            tour[i+1:j+1] = reversed(tour[i+1:j+1])
                            improved = True
                            break
                        else:
                            if gain < best_gain:
                                best_gain = gain
                                best_move = (i, j)
                if improved and first_improvement:
                    break
            if (not first_improvement) and best_move is not None:
                i, j = best_move
                tour[i+1:j+1] = reversed(tour[i+1:j+1])
                improved = True
        return tour


    # ------------- GRASP + 2-opt runner -------------
    def run_grasp_with_twoopt(
        self,
        n_iter=50,
        alpha_city=0.30,
        alpha_pos=0.00,
        seed=0,
        start=None,
        rcl_city_len=None,
        rcl_pos_len=None,
        first_improvement=True,
    ):
        """
        Run GRASP(Insertion) n_iter times; 2-opt each solution; collect (before, after) costs.
        Returns: best_tour, best_cost, xs_before, ys_after
        """
        rng = random.Random(seed)
        xs_before, ys_after = [], []
        best_tour, best_cost = None, float('inf')

        for _ in range(n_iter):
            s = rng.choice(self.cities) if start is None else start
            tour0 = self.getTour_GRASPedInsertion(
                s,
                alpha_city=alpha_city,
                alpha_pos=alpha_pos,
                seed=rng.randint(0, 10**9),
                rcl_city_len=rcl_city_len,
                rcl_pos_len=rcl_pos_len,
            )
            # DEBUG
            if not self.isFeasible(tour0):
                print(f"DEBUG: infeasible tour0 at iteration {it}, length={len(tour0)}")

            cost0 = self.tour_cost(tour0)
            tour1 = self.makeTwoOpt(tour0, first_improvement=first_improvement)

            # DEBUG
            if not self.isFeasible(tour1):
                print(f"DEBUG: infeasible tour1 at iteration {it}, length={len(tour1)}")

            cost1 = self.tour_cost(tour1)

            xs_before.append(cost0)
            ys_after.append(cost1)

            if cost1 < best_cost:
                best_cost = cost1
                best_tour = tour1
        if not self.isFeasible(best_tour):
            print("DEBUG: final best_tour is infeasible!")

        return best_tour, best_cost, xs_before, ys_after


    # --------- Simulated Annealing -------------
    # Simulated Annealing (SA) was tested as an alternative for large instances
    # where 2-opt is too slow. However, it failed to improve the Outlier Insertion
    # solution consistently and is therefore not used in the final experiments.

    def simulated_annealing(
            self,
            tour,
            initial_temp: float = 5000.0,
            cooling_rate: float = 0.99,
            max_iter: int | None = None,
            min_temp: float = 0.00000001,
            seed: int | None = None,
            stagnation_limit: int | None = None,
    ):
        """
        Simulated Annealing (SA) using 2-opt neighborhood (safe mode: recompute costs instead of using delta).
        """

        import math, random

        n = len(tour)
        if max_iter is None:
            max_iter = 1500000 * n

        rng = random.Random(seed)

        # Current solution and its cost
        current = tour[:]
        current_cost = self.tour_cost(current)

        # Best solution found so far
        best = current[:]
        best_cost = current_cost

        # Initial temperature
        T = float(initial_temp)
        it = 0
        no_improve_moves = 0

        while it < max_iter and T > min_temp and (stagnation_limit is None or no_improve_moves < stagnation_limit):

            # Pick random 2-opt move
            i = rng.randrange(0, n - 1)
            j = rng.randrange(i + 1, n)

            # Skip invalid moves
            if (i == 0 and j == n - 1) or (j == i + 1):
                continue

            # Make a copy of current tour and apply move
            new_tour = current[:]
            new_tour[i + 1:j + 1] = list(reversed(new_tour[i + 1:j + 1]))

            # Compute new cost from scratch
            new_cost = self.tour_cost(new_tour)
            delta = new_cost - current_cost

            # Acceptance rule
            accept = False
            if delta < 0:
                accept = True
            else:
                if rng.random() < math.exp(-delta / T):
                    accept = True

            if accept:
                current = new_tour
                current_cost = new_cost

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost
                    no_improve_moves = 0
                else:
                    no_improve_moves += 1
            else:
                no_improve_moves += 1

            # Cool down
            T *= cooling_rate
            it += 1

            # Debug log
            if it < 100:  # first 100 iterations
                print(f"iter={it}, T={T:.2f}, current_cost={current_cost:.2f}, best_cost={best_cost:.2f}")
            elif it % 1000 == 0:  # then every 1000 iterations
                print(f"iter={it}, T={T:.2f}, current_cost={current_cost:.2f}, best_cost={best_cost:.2f}")

        return best

    def makeTwoOpt_fast(self, tour, k=20, max_iter=1000):
        """
        Fast 2-opt using k-nearest neighbors instead of all pairs.

        Parameters
        ----------
        tour : list[int]
            Initial tour to improve.
        k : int
            Number of nearest neighbors to check for each city.
        max_iter : int
            Maximum number of improvement iterations.

        Returns
        -------
        list[int]
            Improved tour.
        """
        n = len(tour)
        tour = tour[:]  # copy
        improved = True
        it = 0

        # Precompute k nearest neighbors for each city
        neighbors = []
        for i in range(n):
            dists = [(j, self.distMatrix[i, j]) for j in range(n) if j != i]
            dists.sort(key=lambda x: x[1])
            neighbors.append([j for j, _ in dists[:k]])

        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(n):
                for j in neighbors[tour[i]]:
                    j_idx = tour.index(j)
                    if abs(i - j_idx) <= 1 or (i == 0 and j_idx == n - 1):
                        continue
                    delta = self._two_opt_gain(tour, i, j_idx)
                    if delta < -1e-12:
                        tour[i + 1:j_idx + 1] = reversed(tour[i + 1:j_idx + 1])
                        improved = True
                        break
                if improved:
                    break
        return tour

    def makeTwoOpt_sampledBest(self, tour, k=200, max_rounds=1000, seed=None):
        """
        Random-sampled best-improvement 2-Opt.
        - In each round, generate k random (i,j) pairs.
        - Select the best improving move (lowest delta).
        - Apply it if it improves the tour; else stop.
        - Stops after max_rounds iterations or no improvement.

        Parameters
        ----------
        tour : list[int]
            Initial tour.
        k : int
            Number of random neighbors to sample per round (default=200).
        max_rounds : int
            Maximum number of rounds (default=1000).
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        list[int]
            Locally improved tour.
        """
        import random
        rng = random.Random(seed)
        n = len(tour)
        tour = tour[:]  # copy
        current_cost = self.tour_cost(tour)

        for _ in range(max_rounds):
            best_delta = 0
            best_move = None

            # sample k random neighbors
            for __ in range(k):
                i = rng.randrange(0, n - 1)
                j = rng.randrange(i + 1, n)
                if (i == 0 and j == n - 1) or (j == i + 1):
                    continue
                delta = self._two_opt_gain(tour, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)

            # if improvement found, apply best move
            if best_move is not None:
                i, j = best_move
                tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                current_cost += best_delta
            else:
                break  # no improvement in this round

        return tour


    def iterated_local_search(self, tour, n_iter=500, perturb_size=5, k=50, seed=None):
        """
        Iterated Local Search (ILS).
        - Start from given tour (e.g. Outlier Insertion).
        - Repeat: perturb, then improve with fast 2-opt.
        - Keep best solution found.

        Parameters
        ----------
        tour : list[int]
            Initial tour (e.g., from Outlier Insertion).
        n_iter : int
            Number of perturbation + local search iterations.
        perturb_size : int
            Number of consecutive cities to shuffle in perturbation.
        k : int
            Number of nearest neighbors checked in fast 2-opt.
        seed : int or None
            Random seed.

        Returns
        -------
        best_tour : list[int]
            Best tour found.
        best_cost : float
            Cost of best tour.
        """
        import random
        rng = random.Random(seed)

        # Initial tour and cost
        current = tour[:]
        current_cost = self.tour_cost(current)
        best = current[:]
        best_cost = current_cost

        for it in range(n_iter):
            # --- Perturbation: shuffle a small segment ---
            perturbed = current[:]
            start = rng.randrange(0, len(perturbed) - perturb_size)
            segment = perturbed[start:start + perturb_size]
            rng.shuffle(segment)
            perturbed[start:start + perturb_size] = segment

            # --- Local search: fast 2-opt ---
            improved = self.makeTwoOpt_fast(perturbed, k=k, max_iter=200)
            improved_cost = self.tour_cost(improved)

            # --- Acceptance: if better, keep it ---
            if improved_cost < best_cost:
                best = improved
                best_cost = improved_cost
                current = improved
                current_cost = improved_cost
                print(f"Iter {it}: New best = {best_cost:.2f}")
            else:
                # Optionally accept even if not better (to escape local optima)
                current = improved
                current_cost = improved_cost

        return best, best_cost

    def plot_tour(self, tour, cost=None, title="TSP Tour"):
        """
        Plot a TSP tour using stored point coordinates.
        """
        xs = [self.points[i].x for i in tour] + [self.points[tour[0]].x]
        ys = [self.points[i].y for i in tour] + [self.points[tour[0]].y]

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        # Normal turu çiz (kırmızı çizgi ve küçük noktalar)
        plt.plot(xs, ys, 'r-', marker='o', markersize=4, label="Tour")

        # Başlangıç şehrini mavi ve büyük göster
        start_x, start_y = self.points[tour[0]].x, self.points[tour[0]].y
        plt.plot(start_x, start_y, 'bo', markersize=10, label="Start city")

        if cost is None:
            cost = self.tour_cost(tour)
        plt.title(f"{title}, Total Distance = {cost:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()
        plt.show()




# ----------------- RUN -----------------
if __name__ == "__main__":

    instFilename = "Instances/Small/berlin52.tsp"
    inst = TSP(instFilename)

    # Nearest Neighbor
    random.seed(3899)
    startPointNN = random.choice(inst.cities)
    tour_nn = inst.getTour_NN(startPointNN)
    print("NN cost:", inst.tour_cost(tour_nn))
    print("NN feasible?", inst.isFeasible(tour_nn))

    # # Outlier Insertion
    tour_oi = inst.getTour_OutlierInsertion(startPointNN)
    print("Outlier Insertion cost:", inst.tour_cost(tour_oi))
    print("OI feasible?", inst.isFeasible(tour_oi))

    # Plot Outlier Insertion tour
    inst.plot_tour(tour_oi, title=f"Outlier Insertion (Cost={inst.tour_cost(tour_oi):.2f})") # plot the tour

    # GRASP + 2-opt demo (quality-based RCLs)
    if inst.nCities <= 200:
        best_tour, best_cost, xs, ys = inst.run_grasp_with_twoopt(
            n_iter=50, alpha_city=0.30, alpha_pos=0.10, seed=125
        )
        print("Best GRASP+2opt cost:", best_cost)
        print("GRASP+2opt feasible?", inst.isFeasible(best_tour))
        inst.plot_tour(best_tour, title=f"GRASP+2opt (Cost={best_cost:.2f})") # plot the tour
        # Scatter plot: before (x) vs after (y)
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, color='blue', alpha=0.7)
        plt.plot([min(xs), max(xs)], [min(xs), max(xs)], color='red', linestyle='--', label="y=x")
        plt.xlabel("Cost before 2-Opt (GRASP tour)")
        plt.ylabel("Cost after 2-Opt")
        plt.title("GRASP + 2-Opt: Before vs After Costs")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # Large instance
        start_tour = inst.getTour_OutlierInsertion(0)
        best_tour, best_cost = inst.iterated_local_search(start_tour, n_iter=200, perturb_size=5, k=50, seed=42)
        print("Best ILS cost:", best_cost)
        print("ILS feasible?", inst.isFeasible(best_tour))
        inst.plot_tour(best_tour, title=f"ILS (Cost={best_cost:.2f})") # plot the tour

#---------------- Plot GRASP + 2-Opt results -----------------

#----------------- NN from all starting positions -----------------
nn_results = inst.run_NN_all_starts()  # returns list of (start_city, cost)
# for start, cost in nn_results:
#     print(f"Start city {start}: NN tour cost = {cost}")

# Optional: minimum, maximum, average
costs = [cost for _, cost in nn_results]
print("NN costs summary:")
print(f"Min: {min(costs)}, Max: {max(costs)}, Avg: {sum(costs)/len(costs):.2f}")
