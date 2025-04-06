# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import base64
import time
from flask import jsonify, session
import base64
import io
from flask import request, jsonify
from PIL import Image
from flask import Blueprint, request, jsonify
import base64
import random
import os, json, pprint
import wtforms
from apps.home import blueprint
from flask import render_template, request, redirect, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask_login import login_required, current_user
from apps import db, config
from apps.models import *
from apps.tasks import *
from apps.authentication.models import Users
from flask_wtf import FlaskForm

class DemographicForm(FlaskForm):
    age = wtforms.IntegerField("Age")
    gender = wtforms.StringField("Gender")
    race = wtforms.StringField("Race")
    ethnicity = wtforms.StringField("Ethnicity")
    country_of_birth = wtforms.StringField("Country of Birth")
    vital_status = wtforms.StringField("Vital Status")


# Default demographic values
DEFAULT_DEMOGRAPHICS = {
    'age': 30,
    'gender': 'Male',
    'race': 'White',
    'ethnicity': 'Not Hispanic or Latino',
    'country_of_birth': 'United States'
}


@blueprint.route('/')
@blueprint.route('/index')
def index():
    # Retrieve cached demographics from session, or use defaults
    demographics = session.get('demographics', DEFAULT_DEMOGRAPHICS)
    return render_template('pages/index.html', segment='index', demographics=demographics)

@blueprint.route('/assessment', methods=['GET', 'POST'])
def demographic_assessment():
    # Get cached demographics from session if available
    cached_data = session.get('demographics')
    if cached_data:
        # Populate the form with existing session data
        form = DemographicForm(data=cached_data)
    else:
        # Instantiate form without any data so defaults are used
        form = DemographicForm()
        form.process()  # Ensures defaults are set

    if form.validate_on_submit():
        session['demographics'] = {
            'age': form.age.data,
            'gender': form.gender.data,
            'race': form.race.data,
            'ethnicity': form.ethnicity.data,
            'country_of_birth': form.country_of_birth.data
        }
        return redirect(url_for('home_blueprint.index'))

    return render_template('pages/assessment.html', form=form)

    return render_template('pages/assessment.html', form=form)
@blueprint.route('/icon_feather')
def icon_feather():
    return render_template('pages/icon-feather.html', segment='icon_feather')

@blueprint.route('/color')
def color():
    return render_template('pages/color.html', segment='color')



@blueprint.route('/picture-taking')
def take_media():
    return render_template('pages/picture-taking.html', segment='picture-taking')

@blueprint.route('/sample_page')
def sample_page():
    return render_template('pages/sample-page.html', segment='sample_page')

@blueprint.route('/typography')
def typography():
    return render_template('pages/typography.html', segment='typography')

@blueprint.route('/api/detect', methods=['POST'])
def detect_skin_disease():
    # this is the place where you capture the image
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    random_boxes = []
    if random.random() > 0.5:  
        for _ in range(random.randint(1, 3)):
            random_boxes.append({
                "x": random.randint(50, 400),
                "y": random.randint(50, 300),
                "width": random.randint(50, 150),
                "height": random.randint(50, 150),
                "label": "Possible Ghost exsists"
            })

    return jsonify({'boxes': random_boxes})



@blueprint.route('/', defaults={'req_path': ''})
@blueprint.route('/<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = 'Users/suchenfeng/Documents/GitHub/HackUSF-2025/website/apps/'

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    return render_template('files.html', files=files)

def getField(column): 
    if isinstance(column.type, db.Text):
        return wtforms.TextAreaField(column.name.title())
    if isinstance(column.type, db.String):
        return wtforms.StringField(column.name.title())
    if isinstance(column.type, db.Boolean):
        return wtforms.BooleanField(column.name.title())
    if isinstance(column.type, db.Integer):
        return wtforms.IntegerField(column.name.title())
    if isinstance(column.type, db.Float):
        return wtforms.DecimalField(column.name.title())
    if isinstance(column.type, db.LargeBinary):
        return wtforms.HiddenField(column.name.title())
    return wtforms.StringField(column.name.title()) 


@blueprint.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():

    class ProfileForm(FlaskForm):
        pass

    readonly_fields = Users.readonly_fields
    full_width_fields = {"bio"}

    for column in Users.__table__.columns:
        if column.name == "id":
            continue

        field_name = column.name
        if field_name in full_width_fields:
            continue

        field = getField(column)
        setattr(ProfileForm, field_name, field)

    for field_name in full_width_fields:
        if field_name in Users.__table__.columns:
            column = Users.__table__.columns[field_name]
            field = getField(column)
            setattr(ProfileForm, field_name, field)

    form = ProfileForm(obj=current_user)

    if form.validate_on_submit():
        readonly_fields.append("password")
        excluded_fields = readonly_fields
        for field_name, field_value in form.data.items():
            if field_name not in excluded_fields:
                setattr(current_user, field_name, field_value)

        db.session.commit()
        return redirect(url_for('home_blueprint.profile'))
    
    context = {
        'segment': 'profile',
        'form': form,
        'readonly_fields': readonly_fields,
        'full_width_fields': full_width_fields,
    }
    return render_template('pages/profile.html', **context)


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None

@blueprint.route('/dashboard')
def dashboard():
    return render_template('pages/dashboard.html')
@blueprint.route('main')
def main():
    return render_template('pages/main.html')

@blueprint.route('/analyze/<image_id>')
def analyze(image_id):
    # 这里会调用模型处理图像
    return render_template('pages/analyze.html', image_id=image_id)

@blueprint.route('/result/<image_id>')
def result(image_id):
    # 渲染模型结果和建议
    return render_template('pages/result.html', image_id=image_id)


@blueprint.route('/error-403')
def error_403():
    return render_template('error/403.html'), 403

@blueprint.errorhandler(403)
def not_found_error(error):
    return redirect(url_for('error-403'))

@blueprint.route('/error-404')
def error_404():
    return render_template('error/404.html'), 404

@blueprint.errorhandler(404)
def not_found_error(error):
    return redirect(url_for('error-404'))

@blueprint.route('/error-500')
def error_500():
    return render_template('error/500.html'), 500

@blueprint.errorhandler(500)
def not_found_error(error):
    return redirect(url_for('error-500'))


# Celery (to be refactored)
@blueprint.route('/tasks-test')
def tasks_test():
    
    input_dict = { "data1": "04", "data2": "99" }
    input_json = json.dumps(input_dict)

    task = celery_test.delay( input_json )

    return f"TASK_ID: {task.id}, output: { task.get() }"


# Custom template filter

@blueprint.app_template_filter("replace_value")
def replace_value(value, arg):
    return value.replace(arg, " ").title()
