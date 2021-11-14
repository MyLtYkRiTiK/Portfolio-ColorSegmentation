import colorsys

import cv2
import numpy as np


class ColorSegmentation:
    """
    Color Segmentation class that validates inputs, find and filter contours and draw them
    """

    def __init__(self, array_image: np.array,
                 rgb_color: str,
                 interest_area: str,
                 draw_convex: str,
                 draw_mask: str,
                 remove_holes: str,
                 ):
        self.image = cv2.imdecode(array_image, cv2.IMREAD_ANYCOLOR)
        self.hsv_image = None
        self.rgb_color = [int(x) / 255 for x in rgb_color.split( )]
        self.lower_bound = self.make_lower_bound( )
        self.upper_bound = self.make_upper_bound( )
        self.area = float(interest_area)
        self.draw_convex = self.str2bool(draw_convex)
        self.draw_mask = self.str2bool(draw_mask)
        self.remove_holes = self.str2bool(remove_holes)
        self.contours = None
        self.hierarchy = None
        self.largest_contours = None
        self.largest_contour_indexes = None
        self.largest_areas = None
        self.inner_contours = []

    @staticmethod
    def str2bool(string):
        return string in ['true', ]

    def make_lower_bound(self):
        bound = colorsys.rgb_to_hsv(*self.rgb_color)
        bound = (bound[0] * 179 - 30, bound[1] * 255 - 130, bound[1] * 255 - 130)
        bound = [max(0, x) for x in bound]
        return np.array([int(x) for x in bound])

    def make_upper_bound(self):
        bound = colorsys.rgb_to_hsv(*self.rgb_color)
        bound = [min(179, bound[0] * 180 + 30), min(255, bound[1] * 255 + 130), min(255, bound[1] * 255 + 130)]
        return np.array([int(x) for x in bound])

    def validate_image(self):
        """
        Check if input image file is image
        """
        try:
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return "Please, check your image"

    def validate_bounds(self):
        """
        Check numeric bounds and area
        """
        try:
            assert 0 < self.area <= 1
            assert len(self.lower_bound) == 3
            assert len(self.upper_bound) == 3
            assert all(np.array([0, 0, 0]) <= self.lower_bound)
            assert all(self.lower_bound <= self.upper_bound)
            assert all(self.upper_bound < np.array([180, 256, 256]))
        except AssertionError:
            return "Please, check your bounds or area"

    def draw_colors(self):
        """
        Function that draws four colors to represent chosen boundaries
        """
        ls = 200
        s_gradient = (np.ones((ls, 1), dtype=np.uint8) *
                      np.linspace(self.lower_bound[1], self.upper_bound[1], ls, dtype=np.uint8))
        v_gradient = np.rot90(np.ones((ls, 1), dtype=np.uint8) *
                              np.linspace(self.lower_bound[1], self.upper_bound[1], ls, dtype=np.uint8))
        h_array = np.linspace(self.lower_bound[0], self.upper_bound[0] + 1, num=4, endpoint=True, dtype=int)
        color_images = []
        for hue in h_array:
            h = hue * np.ones((ls, ls), dtype=np.uint8)
            hsv_color = cv2.merge((h, s_gradient, v_gradient))
            color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            encoded = cv2.imencode(".jpg", color)[1]
            color_images.append(encoded)
        return color_images

    @staticmethod
    def findGreatestContour(contours, hierarchy, image_area, fraction):
        """
        Function that finds contours of expected size
        :param contours: all contours
        :param hierarchy: hierarchy of contours
        :param image_area: area of all image
        :param fraction: expected fraction of the object of interest
        :return: contours of expected size, their indexes in list of all contours and their areas
        """
        largest_contours = []
        largest_areas = []
        min_area = image_area * fraction
        largest_contour_indexes = []
        i = 0
        total_contours = len(contours)
        while i < total_contours:
            area = cv2.contourArea(contours[i])
            if (area > min_area) and hierarchy[0][i][3] == -1:
                largest_contours.append(contours[i])
                largest_areas.append(area)
                largest_contour_indexes.append(i)
            i += 1

        return largest_contours, largest_contour_indexes, largest_areas

    def make_contours(self):
        """
        Function that finds all contours
        """
        mask = cv2.inRange(self.hsv_image, self.lower_bound, self.upper_bound)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        self.contours, self.hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def make_inner_holes(self):
        """
        Function that finds inner holes of expected size and doesn't look at small holes
        For now holes size is static
        """
        holes_area = 0.05
        for i, (cnt, ind, area) in enumerate(zip(
                self.largest_contours,
                self.largest_contour_indexes,
                self.largest_areas)):
            inner_contours_ind = np.where(self.hierarchy[:, 3] == 0)[0]
            inner_contours = [self.contours[x] for x in inner_contours_ind]
            for inner_cnt in inner_contours:
                inner_area = cv2.contourArea(inner_cnt)
                if inner_area > area * holes_area:
                    self.inner_contours.append(inner_cnt)

    def make_color_segmentation(self):
        """
        Main function to find and draw contours
        :return: encoded image and None for error
        """
        self.make_contours( )

        self.largest_contours, self.largest_contour_indexes, self.largest_areas = self.findGreatestContour(
            self.contours, self.hierarchy, self.hsv_image.shape[0] * self.hsv_image.shape[1], self.area)

        if len(self.largest_contour_indexes) > 0:
            self.hierarchy = self.hierarchy.squeeze( )
        else:
            return None, "There is no color of interest of expected size"
        alpha = 0.3
        BLUE = (255, 190, 0)
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)

        output = cv2.cvtColor(self.hsv_image, cv2.COLOR_HSV2BGR)
        overlay = output.copy( )

        if self.remove_holes:
            self.make_inner_holes( )
            [cv2.drawContours(output, x, -1, BLUE, 3) for x in self.inner_contours]
        if self.draw_mask:
            cv2.fillPoly(overlay, self.largest_contours + self.inner_contours, YELLOW)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        if self.draw_convex:
            [cv2.polylines(output, [cv2.convexHull(x, False)], True, GREEN, 3) for x in self.largest_contours]
        [cv2.drawContours(output, x, -1, BLUE, 3) for x in self.largest_contours]

        encoded = cv2.imencode(".jpg", output)[1]
        return encoded, None
