import json
import random
import copy
import time

with open('dataset.json', 'r') as file:
    pieces = json.load(file)

#------ utiles :) -----#

def rotate_piece(piece):
    return {
        'haut': piece['gauche'],
        'droite': piece['haut'],
        'bas': piece['droite'],
        'gauche': piece['bas'],
        'ID': piece['ID']
    }

# sauvegarde le meilleur individu dans un json à afficher avec show_puzzle.py
def save_best_individual(best_individual, filename='best_solution.json'):
    with open(filename, 'w') as f:
        json.dump(best_individual, f)

def classify_pieces(pieces):
    corners = []
    edges = []
    interiors = []

    for piece in pieces:
        zero_count = list(piece.values()).count(0)
        if zero_count == 2:
            corners.append(piece)
        elif zero_count == 1:
            edges.append(piece)
        else:
            interiors.append(piece)

    return corners, edges, interiors

#------ scoring -----#
#minimisation -> peu importe le puzzle : fitness = 0 c'est l'optimal
def evaluation(board):
    fitness = 0
    size = len(board)

    for i in range(size):
        for j in range(size):
            # Horizontale
            if j + 1 < size:
                if board[i][j]['droite'] != board[i][j + 1]['gauche'] or (board[i][j]['droite'] == 0 and board[i][j + 1]['gauche'] == 0):
                    fitness += 1

            # Verticale
            if i + 1 < size:
                if board[i][j]['bas'] != board[i + 1][j]['haut'] or (board[i][j]['bas'] == 0 and board[i + 1][j]['haut'] == 0):
                    fitness += 1

    # 2x2
    for i in range(size - 1):
        for j in range(size - 1):
            top_left = board[i][j]
            top_right = board[i][j + 1]
            bottom_left = board[i + 1][j]
            bottom_right = board[i + 1][j + 1]

            if top_left['droite'] != top_right['gauche']:
                fitness += 1
            if top_left['bas'] != bottom_left['haut']:
                fitness += 1
            if top_right['bas'] != bottom_right['haut']:
                fitness += 1
            if bottom_left['droite'] != bottom_right['gauche']:
                fitness += 1

    return fitness

#------ mutations -----#

# mélanger les pièces de la grille
def scramble_mutation(board):

    pieces = [piece for row in board for piece in row if piece is not None]
    random.shuffle(pieces)

    corner_piece = board[0][0]

    idx = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if (i, j) != (0, 0):
                board[i][j] = pieces[idx]
                idx += 1
    board[0][0] = corner_piece
    return board

# inverse les lignes de la grille
def invert_row_mutation(board):
    x = random.randint(0, len(board) - 1)
    board[x] = board[x][::-1]
    return board

# inverse les colonnes de la grille
def invert_column_mutation(board):
    y = random.randint(0, len(board) - 1)
    for row in board:
        row[y], row[-y-1] = row[-y-1], row[y]
    return board

# inverse une région 2x2 de la grille
def invert_region_mutation(board):
    region_size = 2
    x, y = random.randint(0, len(board) - region_size), random.randint(0, len(board) - region_size)
    region = [row[y:y + region_size] for row in board[x:x + region_size]]
    for i in range(region_size):
        region[i] = region[i][::-1]
    for i in range(region_size):
        for j in range(region_size):
            board[x + i][y + j] = region[i][j]
    return board

# échange deux une région 2x2 de la grille
def swap_region_mutation(board):
    size = len(board)

    x1, y1 = random.randint(0, size - 2), random.randint(0, size - 2)
    x2, y2 = random.randint(0, size - 2), random.randint(0, size - 2)

    region1 = [board[x1][y1:y1 + 2], board[x1 + 1][y1:y1 + 2]]
    region2 = [board[x2][y2:y2 + 2], board[x2 + 1][y2:y2 + 2]]

    # échange les régions
    board[x1][y1:y1 + 2], board[x1 + 1][y1:y1 + 2] = region2[0], region2[1]
    board[x2][y2:y2 + 2], board[x2 + 1][y2:y2 + 2] = region1[0], region1[1]

    return board


# rotation d'une region
def rotate_region_mutation(board):
    size = len(board)
    # coin supérieur gauche d'une région 2x2
    x = random.randint(0, size - 2)
    y = random.randint(0, size - 2)

    region = [
        [board[x][y], board[x][y + 1]],
        [board[x + 1][y], board[x + 1][y + 1]]
    ]

    rotated_region = [
        [region[1][0], region[0][0]],
        [region[1][1], region[0][1]]
    ]

    board[x][y], board[x][y + 1] = rotated_region[0][0], rotated_region[0][1]
    board[x + 1][y], board[x + 1][y + 1] = rotated_region[1][0], rotated_region[1][1]

    return board

# roation d'une piece de la grille
def rotate_piece_mutation(board):
    x = random.randint(0, len(board) - 1)
    y = random.randint(0, len(board) - 1)


    num_rotations = random.randint(0, 3)  # 0 à 3 rotations (0°, 90°, 180°, 270°)
    for _ in range(num_rotations):
        board[x][y] = rotate_piece(board[x][y])

    return board

# échange deux pièces de la grille
def swap_piece_mutation(board):
    size = len(board)
    x1, y1 = random.randint(0, size - 1), random.randint(0, size - 1)
    x2, y2 = random.randint(0, size - 1), random.randint(0, size - 1)

    piece1 = board[x1][y1]
    piece2 = board[x2][y2]

    if sum(v == 0 for v in piece1.values()) == sum(v == 0 for v in piece2.values()):
        board[x1][y1], board[x2][y2] = board[x2][y2], board[x1][y1]

    return board

# fonction de mutation avec les différentes mutations choisie aléatoirement selon un poids
def mutate(board):

    #mutations associés à un poids -> weight
    mutations = [
        (rotate_piece_mutation, 3),
        (rotate_region_mutation, 1),
        (swap_piece_mutation, 3),
        (swap_region_mutation, 2),
        (invert_region_mutation, 1), 
        (invert_column_mutation, 1), 
        (invert_row_mutation, 1),
        (scramble_mutation, 0.2)
    ]
    # poids total
    total_weight = sum(w for _, w in mutations)

    # randomise le nb de mutations à appliquer (pour avoir plus de diversité en cas d'optimum local)
    num_mutations = random.randint(1, 5)

    # plus le poids d'une mutation est élevé plus elle a de chance d'être appliqué
    for _ in range(num_mutations):
        rand_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        for mutation, weight in mutations:
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                mutation_func = mutation
                break

        # applique la mutation
        board = mutation_func(board)

    return board

#------ croisement -----#

def croisement(parent1, parent2):
    size = len(parent1)
    region_size = random.randint(1, size - 1)

    x = random.randint(0, size - region_size)
    y = random.randint(0, size - region_size)

    # croisement region exchange décrit par niang en utilisant l'ID pour ne pas faire de duplication de pièces
    def region_exchange(p1, p2):
        child = copy.deepcopy(p1)
        region_p2 = []

        ids_in_region_p2 = set()
        for i in range(region_size):
            for j in range(region_size):
                piece = p2[x + i][y + j]
                region_p2.append(piece)
                ids_in_region_p2.add(piece['ID'])

        for i in range(size):
            for j in range(size):
                if child[i][j]['ID'] in ids_in_region_p2:
                    child[i][j] = None

        idx = 0
        for i in range(region_size):
            for j in range(region_size):
                child[x + i][y + j] = region_p2[idx]
                idx += 1

        used_ids = {piece['ID'] for row in child for piece in row if piece is not None}
        missing_pieces = [piece for piece in pieces if piece['ID'] not in used_ids]
        idx = 0
        for i in range(size):
            for j in range(size):
                if child[i][j] is None:
                    child[i][j] = missing_pieces[idx]
                    idx += 1

        return child

    child1 = region_exchange(parent1, parent2)
    child2 = region_exchange(parent2, parent1)

    return child1, child2

#------ selection -----#

def tournament_selection(population, fitness_scores, k=3):
    selected = random.sample(list(zip(population, fitness_scores)), k)
    return min(selected, key=lambda x: x[1])[0]

#------ initialisation -----#

def initialisation(pieces, size):
    corners, edges, interiors = classify_pieces(pieces)
    population = []

    def place_corners(board, corners):
        size = len(board)
        if len(corners) < 4:
            raise ValueError("pas assez de pieces d'angles dans le jdd")

        board[0][0] = corners.pop(random.randint(0, len(corners) - 1))
        board[0][size-1] = corners.pop(random.randint(0, len(corners) - 1))
        board[size-1][0] = corners.pop(random.randint(0, len(corners) - 1))
        board[size-1][size-1] = corners.pop(random.randint(0, len(corners) - 1))

    def place_edges(board, edges):
        size = len(board)
        if len(edges) < 4 * (size - 2):
            raise ValueError("pas assez de pieces de côtés dans le jdd")

        for i in range(1, size - 1):
            board[0][i] = edges.pop(random.randint(0, len(edges) - 1))
            board[size - 1][i] = edges.pop(random.randint(0, len(edges) - 1))
            board[i][0] = edges.pop(random.randint(0, len(edges) - 1))
            board[i][size - 1] = edges.pop(random.randint(0, len(edges) - 1))


    def place_interiors(board, interiors):
        size = len(board)
        if len(interiors) < (size - 2) * (size - 2):
            raise ValueError("pas assez de pieces d'intérieure dans le jdd")

        for i in range(1, size - 1):
            for j in range(1, size - 1):
                board[i][j] = interiors.pop(random.randint(0, len(interiors) - 1))


    for _ in range(size):
        board = [[None for _ in range(5)] for _ in range(5)]
        place_corners(board, corners.copy())
        place_edges(board, edges.copy())
        place_interiors(board, interiors.copy())
        population.append(board)

    return population

#------ réparation -----#

def reparation(board):
    for _ in range(10): 
        board = mutate(board)
    return board


#------ algo génétique avec les paramètres -----#

def genetic_algorithm(pieces, population_size=500, generations=3000, mutation_rate=0.7, elite_ratio=0.2):
    start_time = time.time()

    population = initialisation(pieces, population_size)
    
    # nb elite à conserver
    elite_count = max(1, int(population_size * elite_ratio))
    
    # meilleur individu et son fitness
    best = None
    best_fitness = float('inf')

    stagnation_seuil = 100
    stagnation_compteur = 0

    # boucle principale
    for generation in range(generations):
        
        # évaluation du fitness de chaque individu 
        fitness_scores = [evaluation(ind) for ind in population]
        pop_with_scores = list(zip(population, fitness_scores))
        pop_with_scores.sort(key=lambda x: x[1])
        current_best_fitness = pop_with_scores[0][1]

        # MAJ du meilleur individu si trouvé
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best = copy.deepcopy(pop_with_scores[0][0])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # selection des élites à conserver
        elites = [copy.deepcopy(ind) for ind, _ in pop_with_scores[:elite_count]]
        new_population = elites.copy()


        # génration nouvelle popu avec croisement
        while len(new_population) < population_size:
            # sélection des parents par tournoi
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)


            child1, child2 = croisement(parent1, parent2)

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            # ajout des enfants à la nouvelle population
            new_population.extend([child1, child2])

        new_population = new_population[:population_size]

        # réparation si stagnation détecté
        if stagnation_compteur >= stagnation_seuil:
            for i in range(elite_count, population_size):
                if evaluation(new_population[i]) == best_fitness:
                    new_population[i] = reparation(new_population[i])
            stagnation_counter = 0

        # MAJ population
        population = new_population

        print(f"Génération {generation}: Meilleur Fitness = {best_fitness}")

        if best_fitness == 0:
            print("Solution optimale trouvée !")
            break

    save_best_individual(best)
    end_time = time.time()
    print(f"\n Temps total d'exécution : {end_time - start_time:.2f} secondes")

    # retourne le meilleur individu trouvé
    return best


# exécution
best_solution = genetic_algorithm(pieces)
