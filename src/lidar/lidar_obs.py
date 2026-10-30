import numpy as np
import math
import matplotlib.pyplot as plt


class lidar:
    def __init__(self, Range, Angle, points_polygon, centers, agent_radius=0.2):
        self.Distance_Range = Range
        self.Angle_Resolution = Angle
        self.points_polygon = points_polygon
        self.agent_radius = agent_radius
        self.scan_array = []
        self.agents = centers

    @staticmethod
    def delta(a, b, c):
        return b ** 2 - 4 * a * c

    def sense_obstacles(self):
        # rotate at each step for one degree
        for angle in np.linspace(0, 2 * math.pi, int(360 / self.Angle_Resolution), False):
            # array to save distances for one angle ( finally we will get the minimum one)
            distances = []
            distances.append(self.Distance_Range)  # if there is no obstacle, we need to return distance_range
            for polygon in self.points_polygon:
                for i in range(-1, len(polygon) - 1):
                    coord_1 = polygon[i]
                    coord_2 = polygon[i + 1]
                    # find coefficient of quadratic equation to find the intersection point between line and circle
                    m2 = (coord_2[1] - coord_1[1]) / (coord_2[0] - coord_1[0])
                    b2 = coord_2[1] - m2 * coord_2[0]
                    x_intersect = (b2) / (math.tan(angle) - m2)
                    y_intersect = math.tan(angle) * x_intersect
                    if x_intersect >= min(coord_1[0], coord_2[0]) and x_intersect <= max(coord_1[0], coord_2[0]) and y_intersect >= min(coord_1[1], coord_2[1]) and y_intersect <= max(coord_1[1], coord_2[1]):  # check if there is a collision
                        if x_intersect * math.cos(angle) >= 0 and y_intersect * math.sin(angle) >= 0:
                            distances.append(np.linalg.norm([x_intersect, y_intersect]))
            for coord in self.agents:
                # find coefficient of quadratic equation to find the intersection point between line and circle
                a, b, c = 1 + math.tan(angle) ** 2, -2 * coord[0] - 2 * math.tan(angle) * coord[1], coord[0] ** 2 + \
                          coord[1] ** 2 - self.agent_radius ** 2
                if lidar.delta(a, b, c) >= 0:  # check if there is a collision
                    x = np.roots([a, b, c])  # getting the roots of equation (x-coordinate)
                    y = math.tan(angle) * x  # getting the y-coordinate of intersection point
                    if x[0] * math.cos(angle) >= 0 and y[0] * math.sin(
                            angle) >= 0:  # check if the intersection point is in the same coordinate as angle
                        distances.append(min(np.linalg.norm([x[0], y[0]]), np.linalg.norm([x[1], y[1]])))
            self.scan_array.append(min(distances))  # take the minimum between all circles to apply occolusion

    def plot(self):
        figure, axes = plt.subplots()
        plt.figure(figsize=(30, 30))
        plt.clf
        count = 0
        for polygon in self.points_polygon:
            polygon.append(polygon[0])  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*polygon)  # create lists of x and y values
            axes.plot(xs, ys)

        for center in self.agents:
            axes.add_artist(plt.Circle((center[0], center[1]), 0.2))
        for angle in np.linspace(0, 2 * math.pi, int(360 / self.Angle_Resolution), False):
            if count % 2 == 0:  # to be more visible, I am showing every 2 angles
                axes.scatter(self.scan_array[count] * math.cos(angle), self.scan_array[count] * math.sin(angle), s=10,
                             color="r")
                axes.plot([0, self.scan_array[count] * math.cos(angle)], [0, self.scan_array[count] * math.sin(angle)],
                          color="g")
            count += 1
        axes.set_aspect('equal', adjustable='box')
        figure.set_size_inches(18.5, 10.5)
      
        plt.show()

# c=lidar(8,1,[(-5, -3), (-3,-2.999),
#    (-4,-4),(-2.5,-3.8),(-2.8,-5),(-6,-6)])
# c.sense_obstacles()
# c.plot()