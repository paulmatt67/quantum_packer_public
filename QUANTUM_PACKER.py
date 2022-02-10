import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from QUBO_SOLVER import QUBO_SOLVER
from RECTANGLE_PACKER import RECTANGLE_PACKER
import pickle
import random
import copy


class QUANTUM_PACKER():

    def __init__(self, instance_name, board_dimensions, pieces, min_num_pieces_per_rectangle, num_qubits, quantum_machine):
        self.instance_name = instance_name
        self.board_dimensions = board_dimensions
        self.pieces = [Polygon(piece) for piece in pieces]
        self.num_pieces = len(self.pieces)
        self.min_num_pieces_per_rectangle = min_num_pieces_per_rectangle
        self.num_qubits = num_qubits
        self.quantum_machine = quantum_machine
        self.alpha = 0.05
        self.phi_set = np.arange(0, 360, 360.0 / 32.0)
        self.theta_set = np.arange(0, 360, 360.0 / 32.0)

    @staticmethod
    def my_polygon(img, shape, color):
        x, y = shape.exterior.xy
        plt.plot(x, y, color=color)

    def center_of_shape(self, shape):
        center = shape.centroid
        return Point(center.x, center.y)

    @staticmethod
    def rotate_vector(vector, angle):
        """
        Rotates a 2d vector clockwise around the origin.
        :param vector: list of the form [x, y]
        :param angle: rotation angle in degrees
        :return: rotated_vector [x',y']
        """
        x = vector[0]
        y = vector[1]
        # angle converted in radians
        theta = angle * math.pi / 180
        return [x*math.cos(theta)-y*math.sin(theta), x*math.sin(theta)+y*math.cos(theta)]

    def rotate_shape(self, shape, rotation_angle, rotation_center=None):
        if str(type(shape)) == '<class \'list\'>':
            shape = MultiPolygon(shape)
        if rotation_center == None:
            rotation_center = self.center_of_shape(shape)
        try:
            rotated_shape = affinity.rotate(shape, rotation_angle, rotation_center)
        except:
            pass
        return rotated_shape

    def translate_shape(self, shape, vector):
        translated_shape = affinity.translate(shape, xoff=vector.x, yoff=vector.y)
        return translated_shape

    def place_shape(self, shape, position, angle):
        placed_shape = self.rotate_shape(shape, angle)
        c = self.center_of_shape(placed_shape)
        translation_vector = Point(position.x - c.x, position.y - c.y)
        placed_shape = self.translate_shape(placed_shape, translation_vector)
        return placed_shape

    def is_intersection(self, shape1, shape2):
        if str(type(shape1)) == '<class \'list\'>':
            shape1 = MultiPolygon(shape1)
        if str(type(shape2)) == '<class \'list\'>':
            shape2 = MultiPolygon(shape2)
        try:
            intersection_area = shape1.intersection(shape2).area
        except:
            try:
                for polygon in list(shape1):
                    if self.is_intersection(polygon, shape2):
                        return True
                return False
            # if shape1 is not iterable (ex. LineString)
            except:
                return False
        if intersection_area >= 1:
            return True
        else:
            return False

    def open_segment(self, A, B):
        u_AB = np.array([B.x - A.x, B.y - A.x])
        u_AB = u_AB / np.linalg.norm(u_AB, 2)
        u_AB = 1e-5 * u_AB
        A_prime = Point(A.x + u_AB[0], A.y + u_AB[1])
        B_prime = Point(B.x - u_AB[0], B.y - u_AB[1])
        return LineString([A_prime, B_prime])

    def is_facing(self, edge1, edge2, shape1, shape2):
        """
        Check if edge1 of shape1 and edge2 of shape2 are facing each other.
        """
        S12 = self.open_segment(edge1, edge2)
        I1 = self.is_intersection(S12, shape1)
        I2 = self.is_intersection(S12, shape2)
        # if the open segment ]e1, e2[ does not intersect any of the two shapes
        if (I1 == True) or (I2 == True):
            return False
        else:
            return True

    def get_edges(self, piece):
        if str(type(piece)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            edges = list(piece.exterior.coords)
            edges = [Point(e) for e in edges]
        else:
            edges = []
            for polygon in piece.geoms:
                edges += self.get_edges(polygon)
        return edges

    def facing_edges(self, piece1, piece2):
        # edges of piece1
        E1 = self.get_edges(piece1)
        # edges of piece2
        E2 = self.get_edges(piece2)
        # compute facing pairs of edges
        F12 = []
        for e1 in E1:
            for e2 in E2:
                if self.is_facing(e1, e2, piece1, piece2):
                    F12.append((e1, e2))
        # if there are no facing edges
        if len(F12) == 0:
            print('WARNING: there are no facing edges between these two pieces')
            print(piece1)
            print(piece2)
            # plot the pieces
            multi_polygon = MultiPolygon([piece1, piece2])
            fig, axs = plt.subplots()
            axs.set_aspect('equal', 'datalim')
            for geom in multi_polygon.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            # plot the edges of the pieces
            edges = E1 + E2
            for edge in edges:
                axs.scatter(edge.x, edge.y, s=5, c='black')
            # add title
            plt.title('WARNING: there are no facing edges !')
            # show the plot
            plt.show()
            pass
        return F12

    def distance(self, e1, e2):
        return e1.distance(e2)

    def distance_between_pieces(self, piece1, piece2):
        distance_sum = 0.0
        num_pairs = 0
        F12 = self.facing_edges(piece1, piece2)
        # for each pair of facing edges
        for f in F12:
            e1 = f[0]
            e2 = f[1]
            distance_sum += self.distance(e1, e2)
            num_pairs += 1
        if num_pairs > 0:
            average_distance = distance_sum / num_pairs
        else:
            print('WARNING: these two pieces have no facing edges ! The distance between them cannot be calculated...')
        return distance_sum / num_pairs

    @staticmethod
    def overlap(piece1, piece2):
        return piece1.intersects(piece2)

    def compute_minimum_distance_between_shapes_at_angle(self, shape1, shape2, phi, show_result=False):
        d_min = np.inf
        u_phi = [np.cos(phi * math.pi / 180), np.sin(phi * math.pi / 180)]
        for orientation_shape2 in self.theta_set:
            rotated_shape2 = self.rotate_shape(shape2, orientation_shape2)
            shapes_overlap = True
            radius = 10.0
            while shapes_overlap == True:
                translation_vector = Point(radius * u_phi[0], radius * u_phi[1])
                placed_shape2 = self.translate_shape(rotated_shape2, translation_vector)
                if self.overlap(shape1, placed_shape2) == False:
                    shapes_overlap = False
                else:
                    radius += 10.0
            d = self.distance_between_pieces(shape1, placed_shape2)
            if d < d_min:
                d_min = d
                best_orientation_2 = orientation_shape2
                best_translation = translation_vector
                best_placed_shape2 = placed_shape2
        if show_result:
            if str(type(shape1)) == '<class \'shapely.geometry.polygon.Polygon\'>':
                list_of_polygons = [shape1, best_placed_shape2]
            else:
                list_of_polygons = [polygon for polygon in shape1.geoms] + [best_placed_shape2]
            self.plot_Multipolygon(MultiPolygon(list_of_polygons), 'd*=' + str(d_min))
        return d_min, best_orientation_2, best_translation, best_placed_shape2

    def minimum_distance_between_shapes_at_angle(self, shape1, shape2, phi, show_result=False):
        # file where to store the these calculations
        storage_file = 'nofit_function_memory_' + self.instance_name + '.p'
        # check if the storage file exists
        try:
            # load the dictionary
            saved_calculations_dict = pickle.load(open(storage_file, 'rb'))
        except:
            # create an empty dictionary
            saved_calculations_dict = {}
            # save the file
            pickle.dump(saved_calculations_dict, open(storage_file, 'wb'))
        # key associated to the calculation needed
        key_calculation = str(shape1) + '; ' + str(shape2) + '; ' + str(phi)
        # if the calculation has already been made
        if key_calculation in saved_calculations_dict.keys():
            # get the results of the calculation
            (d_min, best_orientation_2, best_translation, best_placed_shape2) = saved_calculations_dict[key_calculation]
        # otherwise
        else:
            # execute the calculation
            d_min, best_orientation_2, best_translation, best_placed_shape2 = self.compute_minimum_distance_between_shapes_at_angle(shape1, shape2, phi)
        # return the results
        return d_min, best_orientation_2, best_translation, best_placed_shape2

    def minimum_distance_between_shapes(self, piece1, piece2, alpha):
        min_distances_per_angle = []
        for phi in self.phi_set:
            d_min, _, _, _ = self.minimum_distance_between_shapes_at_angle(piece1, piece2, phi)
            min_distances_per_angle.append(d_min)
        return np.percentile(min_distances_per_angle, alpha)

    def identical_pieces_lower_index_dict(self):
        # for each piece index, determines which list of pieces of strictly lower index are the same
        identical_pieces_dict = {}
        for i in range(self.num_pieces):
            pieces_identical_to_i = []
            for j in range(i):
                if self.pieces[i].equals(self.pieces[j]):
                    pieces_identical_to_i.append(j)
            identical_pieces_dict.update({i: pieces_identical_to_i})
        return identical_pieces_dict

    def compute_distance_matrix(self, alpha=0.05):
        # file where to store the distance matrix
        storage_file = 'distance_matrix_' + self.instance_name + '.p'
        # check if the distance matrix has already been computed and saved locally
        try:
            D = pickle.load(open(storage_file, 'rb'))
            print('The distance matrix and nofit functions have already been pre-computed and will be directly loaded')
        except:
            print('The distance matrix and nofit functions need to be pre-computed. Please wait...')
            num_shapes = len(self.pieces)
            # for each piece index, determines which list of pieces of strictly lower index are the same
            identical_pieces_dict = self.identical_pieces_lower_index_dict()
            # compute distance matrix avoiding recomputing distances between identical pieces
            D = np.zeros((num_shapes, num_shapes))
            computed_distances = []
            for i in range(num_shapes-1):
                pieces_identical_to_i = identical_pieces_dict[i]
                if len(pieces_identical_to_i) == 0:
                    for j in range(i, num_shapes):
                        pieces_identical_to_j = identical_pieces_dict[j]
                        if len(pieces_identical_to_j) == 0:
                            d_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j], alpha)
                            distance_key = str(i) + '-' + str(j)
                            computed_distances.append(distance_key)
                        else:
                            first_piece_identical_to_j = min(pieces_identical_to_j)
                            if (first_piece_identical_to_j < j):
                                distance_key = str(i) + '-' + str(first_piece_identical_to_j)
                                if distance_key in computed_distances:
                                    d_ij = D[i, first_piece_identical_to_j]
                                else:
                                    d_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j], alpha)
                                    computed_distances.append(distance_key)
                            else:
                                d_ij = self.minimum_distance_between_shapes(self.pieces[i], self.pieces[j], alpha)
                                distance_key = str(i) + '-' + str(j)
                                computed_distances.append(distance_key)
                        # store d_ij in the distance matrix
                        D[i, j] = d_ij
                        D[j, i] = d_ij
                else:
                    first_piece_identical_to_i = min(pieces_identical_to_i)
                    D[i, :] = D[first_piece_identical_to_i, :]
                    D[:, i] = D[i, :]
            # save the distance matrix
            pickle.dump(D, open(storage_file, 'wb'))
            print('Computation done! Saved matrix in ' + storage_file)
        return D

    def decode_TSP_solution(self, results_dictionary):
        num_cities = int(math.sqrt(len(results_dictionary.keys())))
        hamiltonian_path = np.zeros(num_cities)
        for var in results_dictionary.keys():
            parsed_variable_name = var.split('_')
            i = int(parsed_variable_name[1])
            p = int(parsed_variable_name[2])
            value = results_dictionary[var]
            if value == 1:
                hamiltonian_path[p-1] = i
        return hamiltonian_path

    @staticmethod
    def merge(p1, p2):
        if str(type(p1)) == '<class \'shapely.geometry.multipolygon.MultiPolygon\'>':
            list_of_polygons = list(p1) + [p2]
            result = MultiPolygon(list_of_polygons)
        else:
            if str(type(p1)) != '<class \'list\'>':
                result = MultiPolygon([p1, p2])
            else:
                result = MultiPolygon(p1 + [p2])
        return result

    def greedy_packer(self, pieces):
        """
        Packs a list of pieces one by one and in the order given by the list.

        :param pieces: list of pieces as Polygons
        :return: all pieces packed as a MultiPolygon
        """
        # number of pieces to pack
        num_pieces = len(pieces)
        # initialize the packed pieces with the first piece
        packed_pieces = copy.deepcopy([pieces[0]])
        # for following piece i to pack
        for i in range(1, num_pieces):
            # the lastly packed piece
            last_piece_packed = pieces[i-1]
            # the piece to pack now
            piece_to_pack = pieces[i]
            # initialize status of packing
            status_packing = 'fail'
            # initialize the minimum distance at which the new pieces can be packed with all previous ones
            d_star = np.inf
            # for every possible rotation angle phi of piece_to_pack around last_piece_packed
            for phi in self.phi_set:
                d_min, theta_2, translation_2, placed_piece_to_pack = self.minimum_distance_between_shapes_at_angle(last_piece_packed, piece_to_pack, phi)
                # if the placement given by the nofit function for the piece to pack is valid (avoids overlapping)
                if self.is_intersection(packed_pieces, placed_piece_to_pack) == False:
                    # if the placement reduces the distance to the last piece packed
                    if d_min < d_star:
                        # record the best distance and the placement for the piece to pack
                        d_star = d_min
                        placed_piece_to_pack_star = placed_piece_to_pack
                        # update the status of packing
                        status_packing = 'success'
                # else, the placement given by the nofit function at angle phi has failed
                else:
                    # keep theta_2 but keep translating the piece in the direction of translation_2 until the piece fits
                    small_translation = Point(translation_2.x / 10.0, translation_2.y / 10.0)
                    placed_piece_to_pack = self.translate_shape(placed_piece_to_pack, small_translation)
                    while self.is_intersection(packed_pieces, placed_piece_to_pack) == True:
                        placed_piece_to_pack = self.translate_shape(placed_piece_to_pack, small_translation)
                    # measure the distance to the last piece packed
                    self.distance_between_pieces(last_piece_packed, placed_piece_to_pack)
                    # if the placement reduces the distance to the last piece packed
                    if d_min < d_star:
                        # record the best distance and the placement for the piece to pack
                        d_star = d_min
                        placed_piece_to_pack_star = placed_piece_to_pack
                        # update the status of packing
                        status_packing = 'success'
            # if the packing if piece i was possible
            if status_packing == 'success':
                # greedily pack the piece to pack with the others according to the best placement found
                packed_pieces = copy.deepcopy(self.merge(packed_pieces, placed_piece_to_pack_star))
            # otherwise, packing is impossible and must be aborted
            else:
                packed_pieces = None
                break
        return packed_pieces

    def plot_Multipolygon(self, multi_polygon, my_title=''):
        # if multi_polygon is just a polygon then make it a list of one polygon
        if str(type(multi_polygon)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            multi_polygon = MultiPolygon([multi_polygon])
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        for geom in multi_polygon.geoms:
            xs, ys = geom.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
        plt.title(my_title)
        plt.show()

    def show_pieces(self):
        num_pieces = len(self.pieces)
        num_pieces_per_row = int(math.sqrt(num_pieces))
        W = 0
        H = 0
        for i, p in enumerate(self.pieces):
            bbox = p.bounds
            minx = bbox[0]
            miny = bbox[1]
            maxx = bbox[2]
            maxy = bbox[3]
            width = maxx - minx
            height = maxy - miny
            if width > W:
                W = width
            if height > H:
                H = height
        cell_spacing = 20
        W += cell_spacing
        H += cell_spacing
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        row = 1
        column = 1
        for i, p in enumerate(self.pieces):
            position = Point([row * W, column * H])
            p_placed = self.place_shape(p, position, 0)
            xs, ys = p_placed.exterior.xy
            axs.fill(xs, ys, alpha=1.0, fc='r', ec='none')
            axs.text(position.x, position.y, 'P'+str(i))
            if column >= num_pieces_per_row:
                column = 1
                row += 1
            else:
                column += 1
        plt.title(self.instance_name + ': ' + str(num_pieces) + ' pieces')
        plt.show()

    def fit_bounding_box(self, multi_polygon, show_result=False):
        """
        Returns the general minimum bounding rectangle that contains the object.
        This rectangle is not constrained to be parallel to the coordinate axes.
        """
        # if multi_polygon is just a polygon then make it a list of one polygon
        if str(type(multi_polygon)) == '<class \'shapely.geometry.polygon.Polygon\'>':
            multi_polygon = MultiPolygon([multi_polygon])
        # if multi_polygon is a list of polygons, then make it a MultiPolygon
        if str(type(multi_polygon)) == '<class \'list\'>':
            multi_polygon = MultiPolygon(multi_polygon)
        # find rectangular bounding box with minimum area
        bounding_box = multi_polygon.minimum_rotated_rectangle
        # show result of fitting a bounding box around the pieces
        if show_result:
            fig, axs = plt.subplots()
            axs.set_aspect('equal', 'datalim')
            xs, ys = bounding_box.exterior.xy
            axs.fill(xs, ys, alpha=0.5, fc='b', ec='none')
            for geom in multi_polygon.geoms:
                xs, ys = geom.exterior.xy
                axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            plt.title('Bounding box')
            plt.show()
        return bounding_box

    def get_bounding_box_dimensions(self, polygon):
        # get coordinates of polygon vertices
        x, y = polygon.exterior.coords.xy
        # get length of bounding box edges
        dimensions = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        return dimensions

    @staticmethod
    def build_EXACT_COVER_objective_function(N, S):
        """
        Given a set of N elements, X={c_1, ..., c_N}, a family of M subsets of X,
        S={S_1, ..., S_M} such that S_i \subset X and \cup_{i=1}^M S_i = X, find a
        subset I of {1, ..., M} such that \cup_{i \in I} S_i = X where S_i \cap S_j = \emptyset
        for i \neq j \in I.

        :param N: number of elements in X
        :param S: list of lists of integers in [1, ..., N]
        :return: list I of indices of the sets selected from S
        """
        # number of subsets in S
        M = len(S)
        # build list of binary variables s_i, i \in {1, ..., M}
        binary_variables = []
        for i in range(1, M + 1):
            binary_variables.append('s_' + str(i))
        # build list of coefficients w_i of the binary variables s_i
        linear_coefficients = []
        for i in range(1, M + 1):
            # coefficient of s_i is w_i = - 2 * |S_i|
            w_i = - 2 * len(S[i-1])
            linear_coefficients.append(w_i)
        # build dictionary of quadratic coefficients
        quadratic_coefficients = {}
        for i in range(1, M + 1):
            for k in range(1, M + 1):
                # coefficient of s_i * s_k is |S_i \cap S_k|
                w_ik = len([e for e in S[i-1] if e in S[k-1]])
                s_i = 's_' + str(i)
                s_k = 's_' + str(k)
                quadratic_coefficients.update({(s_i, s_k): w_ik})
        return binary_variables, linear_coefficients, quadratic_coefficients

    def pack_rectangles(self, rectangles, rectangle_names, bins):
        my_rect_packer = RECTANGLE_PACKER(rectangles, rectangle_names, bins)
        rectangles_arrangement = my_rect_packer.solve(show_result=True)
        return rectangles_arrangement

    def unpack_rectangles(self, rectangles_arrangement, R):
        # initialize the layout (list of polygons)
        layout = []
        # plot the pieces
        for rectangle_id, value in rectangles_arrangement.items():
            # position of the rectangle in the board
            x, y, w, h, b = value[0], value[1], value[2], value[3], value[4]
            rectangle = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            # pieces packed in a bounding box
            packed_pieces = R[rectangle_id][0]
            bounding_box = R[rectangle_id][1]
            # determines the geometric transformation (translation and rotation) that maps the bounding box to the rectangle
            position_bounding_box = self.center_of_shape(bounding_box)
            position_rectangle = self.center_of_shape(rectangle)
            translation = Point(
                (position_rectangle.x - position_bounding_box.x, position_rectangle.y - position_bounding_box.y))
            superposition_ratio_star = 0.0
            for angle in range(360):
                placed_bounding_box = self.place_shape(bounding_box, position_rectangle, angle)
                superposition_ratio = rectangle.intersection(placed_bounding_box).area / rectangle.area
                if superposition_ratio > superposition_ratio_star:
                    superposition_ratio_star = superposition_ratio
                    angle_star = angle
                if superposition_ratio_star > 0.999:
                    break
            # move the packed pieces using this transformation
            packed_pieces = self.rotate_shape(packed_pieces, angle_star, position_bounding_box)
            packed_pieces = self.translate_shape(packed_pieces, translation)
            # if packed_pieces is just a polygon then make it a list of one polygon
            if str(type(packed_pieces)) == '<class \'shapely.geometry.polygon.Polygon\'>':
                packed_pieces = MultiPolygon([packed_pieces])
            # adds the packed pieces to the layout
            layout += list(packed_pieces)
        return layout

    def local_optimization(self, layout, container):
        """
        Moves all pieces of the layout to the left, while avoiding piece overlapping and going out of the container.

        :param layout: list of pieces
        :param container: a polygon
        :return: optimized layout
        """
        print('    running local optimization...')
        # prioritized list of unitary vectors for translating to the left
        prioritized_directions = []
        for theta in range(-math.pi, -math.pi / 2, math.pi / 64.0):
            prioritized_directions.append(Point([math.cos(theta), math.sin(theta)]))
            prioritized_directions.append(Point([math.cos(theta), -math.sin(theta)]))
        print(prioritized_directions)
        is_optimal = False
        while is_optimal == False:
            is_optimal = True
            for i, piece in enumerate(layout):
                # list of pieces without i
                other_pieces = layout
                del other_pieces[i]
                # move the piece towards the origin as much as possible, while avoiding overlapping with the other ones
                keep_moving_piece = True
                while keep_moving_piece:
                    # test if we can move the piece in one of the allowed directions and if so do it
                    keep_moving_piece = False
                    # for each allowed direction
                    for vector in prioritized_directions:
                        # translate the piece
                        translated_piece = self.translate_shape(piece, vector)
                        # if the piece is within the container
                        if container.contains(translated_piece):
                            # check if the translated piece intersects another one
                            intersection_test = False
                            for other_piece in other_pieces:
                                if self.is_intersection(translated_piece, other_piece):
                                    intersection_test = True
                                    break
                            # and if the translated piece doesn't intersect another one
                            if not intersection_test:
                                # translate the piece
                                piece = translated_piece
                                # keep moving it
                                keep_moving_piece = True
                                # the layout is still not optimal
                                is_optimal = False
                                # don't look at the remaining directions
                                break
                # replace moved pieced i of the layout
                other_pieces[i:i] = [piece]
                layout = other_pieces
        # return optimal layout
        return layout

    def is_in_container(self, multipolygon, container):
        # bounds of the multipolygon
        bounds = list(multipolygon.bounds)
        multipolygon_x_min = round(bounds[0], 3)
        multipolygon_y_min = round(bounds[1], 3)
        multipolygon_x_max = round(bounds[2], 3)
        multipolygon_y_max = round(bounds[3], 3)
        # bounds of the container
        bounds = list(container.bounds)
        x_min = round(bounds[0], 3)
        y_min = round(bounds[1], 3)
        x_max = round(bounds[2], 3)
        y_max = round(bounds[3], 3)
        # test if the multipolygon is in the container
        if (multipolygon_x_min >= x_min) and (multipolygon_y_min >= y_min) and (multipolygon_x_max <= x_max) and (multipolygon_y_max <= y_max):
            return True
        else:
            return False

    def generate_partitions(self, num_partitions, min_cardinality, max_cardinality):
        """
        Generates a list of random partitions of the set {0, 1, ..., num_pieces-1}.

        :param num_partitions: number of partitions generates
        :param max_cardinality: maximum cardinality of elements in the partitions
        :return: list of partitions, each partition itself being a list of strings of the form '0-2-3-4-5-9'
        """
        partitions = []
        support = []
        for i in range(num_partitions):
            # initialize a new partition
            partition = []
            # initialize list of piece indices
            Q = list(range(self.num_pieces))
            while len(Q) > 0:
                # generate a random integer k between 1 and max_cardinality
                k = random.randint(min_cardinality, max_cardinality)
                # select randomly k elements from Q without replacement
                try:
                    P = random.sample(Q, k)
                # unless Q has size less than k
                except:
                    # take all pieces left in Q
                    P = Q
                # sort the indices in P
                P.sort()
                # add the P as element to the partition
                partition.append(P)
                # add P to the support, if P is not already in the support
                if P not in support:
                    support.append(P)
                # remove P from Q
                Q = [e for e in Q if e not in P]
            # add the partition to the list
            partitions.append(partition)
        # return the list of partitions
        return partitions, support

    def modulo_key(self, key):
        """
        Takes a key for a set of pieces, e.g. '0-4-12-15' and  replaces each piece index
        by the minimum index of an identical piece.

        :param key:
        :return: string
        """
        indices = key.split('-')
        indices = [int(s) for s in indices]
        minimum_indices = []
        identical_pieces_dict = self.identical_pieces_lower_index_dict()
        for index in indices:
            try:
                minimum_index = min(identical_pieces_dict[index])
            except:
                minimum_index = index
            minimum_indices.append(str(minimum_index))
        modulo_key = '-'.join(minimum_indices)
        return modulo_key

    def is_valid_path(self, result):
        num_vertices = int(math.sqrt(len(list(result.keys()))))
        x_matrix = np.zeros((num_vertices, num_vertices))
        for binary_variable in result.keys():
            strings = binary_variable.split('_')
            i = int(strings[1])
            p = int(strings[2])
            x_matrix[i-1, p-1] = result[binary_variable]
        for p in range(num_vertices):
            if np.sum(x_matrix[:, p]) != 1.0:
                return False
        for i in range(num_vertices):
            if np.sum(x_matrix[i, :]) != 1.0:
                return False
        return True

    def required_bin_length(self, layout):
        """
        Maximum x coordinate of a polygon in the layout
        :param layout: list of polygons
        :return: maximum x coordinate
        """
        return max([max(polygon.exterior.xy[0]) for polygon in layout])

    def build_TSP_objective_function(self, W):
        N = W.shape[0]
        # maximum of absolute distances
        Wmax = np.abs(W).max()
        # Penalty coefficient
        A = math.ceil(Wmax) + 1.0
        # build list of binary variables
        binary_variables = []
        for i in range(1, N+1):
            for p in range(1, N+1):
                binary_variables.append('x_'+str(i)+'_'+str(p))
        # build list of coefficients of the binary variables
        linear_coefficients = []
        for i in range(1, N+1):
            for p in range(1, N+1):
                # coefficient of x_i_p is -4A
                linear_coefficients.append(-4.0 * A)
        # build dictionary of quadratic coefficients
        quadratic_coefficients = {}

        for i in range(1, N+1):
            for j in range(1, N+1):
                for p in range(1, N):
                    var1 = 'x_'+str(i)+'_'+str(p)
                    var2 = 'x_'+str(j)+'_'+str(p+1)
                    quadratic_coefficients.update({(var1, var2): W[i-1, j-1]})

        for p in range(1, N+1):
            for i in range(1, N+1):
                for i_prime in range(1, N+1):
                    var1 = 'x_' + str(i) + '_' + str(p)
                    var2 = 'x_' + str(i_prime) + '_' + str(p)
                    quadratic_coefficients.update({(var1, var2): A})

        for i in range(1, N+1):
            for p in range(1, N+1):
                for p_prime in range(1, N+1):
                    var1 = 'x_' + str(i) + '_' + str(p)
                    var2 = 'x_' + str(i) + '_' + str(p_prime)
                    if (var1, var2) in quadratic_coefficients:
                        quadratic_coefficients.update({(var1, var2): A + quadratic_coefficients[(var1, var2)]})
                    else:
                        quadratic_coefficients.update({(var1, var2): A})

        return binary_variables, linear_coefficients, quadratic_coefficients

    def solve(self, num_trials=10):
        # initialize the bin and the larger bin
        W = self.board_dimensions[0]
        H = self.board_dimensions[1]
        bins = [(W, H)]
        larger_bins = [(2*W, H)]
        # container and larger_container (bin and larger_bin as Polygon)
        container = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
        larger_container = Polygon([(0, 0), (2*W, 0), (2*W, H), (0, H)])
        # Compute the matrix of distances between pieces
        D = self.compute_distance_matrix(alpha=0.05)
        print(D)
        # Generate num_trials partitions of the set of pieces and their support (elements that belong to their union)
        min_num_pieces_per_rectangle = self.min_num_pieces_per_rectangle
        max_num_pieces_per_rectangle = int(math.floor(math.sqrt(self.num_qubits)))
        partitions, partitions_support = self.generate_partitions(num_trials, min_num_pieces_per_rectangle, max_num_pieces_per_rectangle)
        # Solve the TSP problem for each set of pieces in partitions_support and store the hamiltonian paths
        print('Solving ' + str(len(partitions_support)) + ' TSP problems...')
        hamiltonian_paths_dict = {}
        for tsp_problem_index, P in enumerate(partitions_support):
            # key for the set of the pieces P
            set_of_pieces_key = '-'.join([str(i) for i in P])
            # modulo key for the set of pieces
            set_of_pieces_modulo_key = self.modulo_key(set_of_pieces_key)
            print('TSP problem #' + str(tsp_problem_index+1) + ' for set of pieces {' + set_of_pieces_key.replace('-', ',') + '} REF:' + set_of_pieces_modulo_key)
            # if the modulo key is already in R
            if set_of_pieces_modulo_key in hamiltonian_paths_dict.keys():
                print('     reusing Hamiltonian path from set ' + set_of_pieces_modulo_key)
                # get the path already computed for the modulo key
                hamiltonian_path = hamiltonian_paths_dict[set_of_pieces_modulo_key]
                # store the result for the key
                hamiltonian_paths_dict.update({set_of_pieces_key: hamiltonian_path})
            else:
                # distance sub-matrix for the pieces in P
                D_P = D[P, :][:, P]
                # Solve the TSP problem for D_P on the selected quantum machine
                binary_variables, linear_coefficients, quadratic_coefficients = self.build_TSP_objective_function(D_P)
                TSP = QUBO_SOLVER(binary_variables, linear_coefficients, quadratic_coefficients, self.quantum_machine)
                # solve the TSP until we get a valid path
                is_valid_path = False
                while is_valid_path != True:
                    result = TSP.resolve()
                    print(result)
                    if self.is_valid_path(result):
                        is_valid_path = True
                        print('     TSP solved !')
                    else:
                        print('     TSP solution is not a valid path, restarting...')
                hamiltonian_path = self.decode_TSP_solution(result).astype(int).tolist()
                # Pack pieces in the order of the hamiltonian path
                ordered_piece_indices_to_pack = [P[i - 1] for i in hamiltonian_path]
                # store the hamiltonian path
                hamiltonian_paths_dict.update({set_of_pieces_key: ordered_piece_indices_to_pack})
                hamiltonian_paths_dict.update({set_of_pieces_modulo_key: ordered_piece_indices_to_pack})
        # Pack all greedily sets of pieces in rectangles in the order of the hamiltonian path
        # and store into the pool of rectangles R (dictionary)
        print('Packing pieces in ' + str(len(partitions_support)) + ' rectangles...')
        R = {}
        for P in partitions_support:
            # key for the set of the pieces P
            set_of_pieces_key = '-'.join([str(i) for i in P])
            # modulo key for the set of pieces
            set_of_pieces_modulo_key = self.modulo_key(set_of_pieces_key)
            print('Packing pieces ' + set_of_pieces_key)
            # if the modulo key is already in R
            if set_of_pieces_modulo_key in R.keys():
                print('     reusing packed pieces from set ' + set_of_pieces_modulo_key)
                # get the packed pieces already computed for the modulo key
                (packed_pieces, bbox, dimensions) = R[set_of_pieces_modulo_key]
                # store the result for the key
                R.update({set_of_pieces_key: (packed_pieces, bbox, dimensions)})
            else:
                # get the hamiltonian path
                ordered_piece_indices_to_pack = hamiltonian_paths_dict[set_of_pieces_key]
                # pack greedily and form a MultiPolygon
                ordered_piece_to_pack = [copy.deepcopy(self.pieces[i]) for i in ordered_piece_indices_to_pack]
                packed_pieces = self.greedy_packer(ordered_piece_to_pack)
                # if the greedy packer was successful
                if packed_pieces is not None:
                    # Fit a (rotated) bounding box around a set of pieces
                    bbox = self.fit_bounding_box(packed_pieces)
                    # Get bounding box dimensions
                    dimensions = self.get_bounding_box_dimensions(bbox)
                    # store in the packed rectangles and in the pool with the key and also the modulo key
                    R.update({set_of_pieces_key: (packed_pieces, bbox, dimensions)})
                    R.update({set_of_pieces_modulo_key: (packed_pieces, bbox, dimensions)})
                    # show how the pieces are packed in a rectangle
                    #print('Showing the result of greedy packer for:')
                    #print(ordered_piece_indices_to_pack)
                    #self.show_layout(packed_pieces, rectangles_dict={set_of_pieces_key: (packed_pieces, bbox, dimensions)})
                # otherwise
                else:
                    print('Failed to pack greedily this list of pieces')
        # Initialize the best layout in case none is found that fits the bin size
        min_length_needed = np.inf
        best_layout_found = None
        # For each partition (trial)
        for t, partition in enumerate(partitions):
            print('Trial ' + str(t + 1))
            try:
                # build the rectangle names (keys of each element in the partition)
                rectangle_names = ['-'.join([str(i) for i in e]) for e in partition]
                # build the list of rectangles containing the packed pieces
                rectangles = [R[rectangle_name][2] for rectangle_name in rectangle_names]
                print('Solving rectangular packing problem...')
                rectangles_arrangement = self.pack_rectangles(rectangles, rectangle_names, bins)
                # if the packing of rectangles is successful
                if rectangles_arrangement is not None:
                    # stop searching for a solution
                    print('Solution found !')
                    layout = self.unpack_rectangles(rectangles_arrangement, R)
                    layout = self.local_optimization(layout, larger_container)
                    self.show_layout(layout, board_dimensions=None, rectangles_dict=rectangles_arrangement)
                    return layout
                else:
                    print('Failed ! Retrying with larger bin...')
                    rectangles_arrangement = self.pack_rectangles(rectangles, rectangle_names, larger_bins)
                    # if the packing of rectangles with the larger bin is successful
                    if rectangles_arrangement is not None:
                        # stop searching for a solution
                        print('Optimizing layout found for the larger bin...')
                        layout = self.unpack_rectangles(rectangles_arrangement, R)
                        # show layout in the larger bin before local optimization
                        #self.show_layout(layout, board_dimensions=larger_bins[0], rectangles_dict=rectangles_arrangement)
                        # show layout in the normal bin after local optimization
                        layout = self.local_optimization(layout, larger_container)
                        self.show_layout(layout, board_dimensions=bins[0], rectangles_dict=rectangles_arrangement)
                        # check if the optimized layout fits the container
                        if self.is_in_container(MultiPolygon(layout), container):
                            print('Solution found ! Optimized layout fits the container.')
                            return layout
                        else:
                            print('Optimized layout does not fit the container.')
                            length_needed = self.required_bin_length(layout)
                            if length_needed < min_length_needed:
                                min_length_needed = length_needed
                                best_layout_found = layout
                    else:
                        print('No layout found for the larger bin.')
                        length_needed = self.required_bin_length(layout)
                        if length_needed < min_length_needed:
                            min_length_needed = length_needed
                            best_layout_found = layout
            except:
                'Trial failed due to impossibility to pack some set of pieces greedily'
        # if no layout is found, return the best one found (with minimum length)
        print('Best layout found requires bin length of ' + str(min_length_needed))
        return best_layout_found

    def show_layout(self, layout, board_dimensions=None, rectangles_dict=None):
        # create a new plot
        fig, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')
        # plot the board
        if board_dimensions == None:
            W = self.board_dimensions[0]
            H = self.board_dimensions[1]
            board_color = 'green'
        else:
            W = board_dimensions[0]
            H = board_dimensions[1]
            board_color = 'orange'
        board = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
        xs, ys = board.exterior.xy
        axs.fill(xs, ys, alpha=1.0, fc=board_color, ec='none')
        # plot the rectangles
        if rectangles_dict is not None:
            for rectangle_name in rectangles_dict.keys():
                (x, y, w, h, b) = rectangles_dict[rectangle_name]
                my_rect = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])
                xs, ys = my_rect.exterior.xy
                axs.fill(xs, ys, alpha=1.0, fc='blue', ec='none')
                bbox_center = self.center_of_shape(my_rect)
                axs.text(bbox_center.x, bbox_center.y, 'R(' + rectangle_name.replace('-', ',') + ')')
        # plot the pieces
        for i, piece in enumerate(layout):
            xs, ys = piece.exterior.xy
            axs.fill(xs, ys, alpha=1.0, fc='r', ec='none')
        plt.title('Layout for ' + self.instance_name)
        plt.show()

