from flask import Blueprint, render_template, redirect, url_for, session
from apps.forms import DemographicForm

blueprint = Blueprint('assessment_blueprint', __name__, url_prefix='')

@blueprint.route('/assessment', methods=['GET', 'POST'])
def demographic_assessment():
    form = DemographicForm()
    if form.validate_on_submit():
        session['has_taken_assessment'] = True
        return redirect(url_for('home_blueprint.index'))
    return render_template('/pages/assessment.html', form=form)