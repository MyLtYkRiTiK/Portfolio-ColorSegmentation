import base64

import numpy
from flask import Flask, request, render_template

from src.colorsegmentation import ColorSegmentation

app = Flask(__name__)


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('mainPage.html')

    if request.method == 'POST':
        request_image = request.files['Image']
        request_lower_bound = request.form['lowerBound']
        request_upper_bound = request.form["upperBound"]
        request_area = request.form["Area"]
        request_convex = request.form["gridConvex"]
        request_segmentation_mask = request.form["gridSegmentation"]
        request_holes = request.form["gridHoles"]
        image_string = request_image.stream.read( )
        array_image = numpy.fromstring(image_string, numpy.uint8)

        color_segmentation = ColorSegmentation(array_image=array_image,
                                               lower_bound=request_lower_bound,
                                               upper_bound=request_upper_bound,
                                               interest_area=request_area,
                                               draw_convex=request_convex,
                                               draw_mask=request_segmentation_mask,
                                               remove_holes=request_holes
                                               )
        bound_validationError = color_segmentation.validate_bounds( )
        imageError = color_segmentation.validate_image( )

        if bound_validationError:
            return render_template('validationErrors.html',
                                   validationError=bound_validationError)
        elif imageError:
            return render_template('validationErrors.html',
                                   validationError=imageError)
        else:
            color_images = color_segmentation.draw_colors( )
            color_images_as_text = [base64.b64encode(color_image) for color_image in color_images]
            color_segmentation_image, segmentation_error = color_segmentation.make_color_segmentation( )
            if segmentation_error:
                return render_template('validationErrors.html',
                                       validationError=segmentation_error)
            else:
                color_segmentation_image_as_text = base64.b64encode(color_segmentation_image)
                return render_template('resultPage.html',
                                       color_images=color_images_as_text,
                                       color_segmentation_image=color_segmentation_image_as_text,
                                       )


if __name__ == '__main__':
    app.run(threaded=True)
