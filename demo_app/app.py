import os
import argparse
import datetime

import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from nsds.common import Params

from modules.face_model import FaceModelWrapper
from modules.utils import encode_based64, encode_base64_from_file, \
    decode_base64, maintain_aspect_ratio_resize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('-p', '--port', type=int, default=5005)
    return parser.parse_args()


IMG_PREFIX = 'data:image/jpg;base64,'

args = parse_args()
params = Params.from_file(args.config_path)
data_dir = params['demo']['data_dir']
face_model = FaceModelWrapper(params)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

default_query_img = encode_base64_from_file(
    'demo_app/imgs/default_400x400.jpg')
default_res_img = encode_base64_from_file(
    'demo_app/imgs/default_200x200.jpg')

app.layout = html.Div([
    dcc.Upload(
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        id='upload-image',
        style={
            'width': '30%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0 auto'
        },
        multiple=False,
        accept='image/*'
    ),
    html.Div([
        html.Div([
            html.Img(
                id='query-image-upload',
                src=IMG_PREFIX + default_query_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '8px',
                    'vertical-align': 'middle'
                }
            ),
        ],
            className='column',
            style={
                'flex': '25%',
                'padding': '200px 0px 50px 300px'}
        ),
        html.Div([
            html.Img(
                id='output-image-1',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-2',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-3',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-4',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-5',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-6',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-7',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-8',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            ),
            html.Img(
                id='output-image-9',
                src=IMG_PREFIX + default_res_img,
                style={
                    'borderWidth': '1px',
                    'borderRadius': '5px',
                    'margin-top': '10px',
                    'margin-right': '10px',
                    'vertical-align': 'middle'}
            )
        ],
            className='column',
            style={
                'flex': '25%',
                'padding': '70px 0px'}
        )],
        className='row',
        style={
            'display': 'flex',
            'flex-wrap': 'wrap',
            'padding': '0 4px'}
    )]
)


def parse_contents(contents):
    return html.Div([
        html.Img(src=contents),
        html.Hr(),
    ])


def call_simvec(img):
    res = face_model.get_similar_images(img)
    res_base64_imgs = []
    if res is None:
        return []
    for r in res:
        uid = r.split('_')[0]
        image_id = r[r.index('_') + 1:]
        img_path = os.path.join(data_dir, uid, image_id)
        img = cv2.imread(img_path)
        img = maintain_aspect_ratio_resize(img, height=200)
        res_base64_imgs.append(IMG_PREFIX + encode_based64(img))
    return res_base64_imgs


@app.callback([Output('query-image-upload', 'src')] +
              [Output(f'output-image-{i}', 'src') for i in range(1, 10)],
              [Input('upload-image', 'contents')])
def update_output(contents):
    global ind
    if contents is None:
        raise dash.exceptions.PreventUpdate
    if contents is not None:
        query_img = decode_base64(contents[contents.index(',') + 1:])
        res = call_simvec(query_img)
        if len(res) == 0:
            res = [IMG_PREFIX + default_res_img] * 9
        query_img = maintain_aspect_ratio_resize(query_img, height=400)
        query_img_base64 = encode_based64(query_img)
        return [IMG_PREFIX + query_img_base64] + [r for r in res]


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', threaded=False, port=args.port)
