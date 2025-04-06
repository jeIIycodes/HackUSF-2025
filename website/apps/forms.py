from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional

class DemographicForm(FlaskForm):
    age = IntegerField("Age", default=30, validators=[DataRequired()])
    gender = SelectField(
        "Gender",
        choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')],
        default='Male',
        validators=[DataRequired()]
    )
    race = StringField("Race", default="White", validators=[Optional()])
    ethnicity = StringField("Ethnicity", default="Not Hispanic or Latino", validators=[Optional()])
    country_of_birth = StringField("Country of Birth", default="United States", validators=[Optional()])
    submit = SubmitField("Submit Assessment")
