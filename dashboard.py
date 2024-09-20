import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Read the dataset
data = pd.read_csv("data.csv")

# Clean data
data.drop(["Unnamed: 32"], inplace=True, axis=1)

# Pre-calculate the correlation matrix
numeric_df = data.select_dtypes(include=[float, int]).dropna(axis='columns')
correlation_matrix = numeric_df.corr()

# Pre-calculate feature statistics (mean, variance, std err)
feature_stats = {
    feature: {
        "mean": data[feature].mean(),
        "variance": data[feature].var(),
        "std_err": data[feature].sem()
    } for feature in data.columns[2:-1]
}

# Dropdown options for selecting features
feature_options = [{'label': feature, 'value': feature} for feature in data.columns[2:-1]]

# Create a copy of the data for logistic regression model
data_copy = data.copy()

# Split the data for training the logistic regression model
X = data_copy.drop(['diagnosis'], axis=1)
y = data_copy['diagnosis'].map({'B': 0, 'M': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train_imputed)
X_test_standardized = scaler.transform(X_test_imputed)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train_standardized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_standardized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Map the numeric labels to class names for the classification report
classification_report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=['Benign', 'Malignant'])

# Create a more readable confusion matrix by mapping 0 to "Benign" and 1 to "Malignant"
conf_matrix = confusion_matrix(y_test, y_pred)

# Format the output strings for display
model_performance = [
    f"Accuracy: {accuracy:.2f}",
    f"Classification Report: {classification_report_dict}",
    f"Confusion Matrix: [['Benign' 'Malignant']\n {conf_matrix[0]}\n {conf_matrix[1]}]"
]

# Create the Dash app
app = Dash(suppress_callback_exceptions=True)

# Layout for the app
app.layout = html.Div([
    html.H1("Breast Cancer Data Dashboard"),
    
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Data Table and Correlation Matrix', value='tab-1'),
        dcc.Tab(label='Feature Details with Correlation to Diagnosis', value='tab-2'),
        dcc.Tab(label='Logistic Regression Performance', value='tab-3'),
    ]),
    
    html.Div(id='tabs-content')
])

# Callback to render content based on selected tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H2("Data Table"),
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": col, "id": col} for col in data.columns],
                data=data.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            ),
            html.H2("Correlation Matrix"),
            dcc.Graph(
                id='correlation-matrix-plot',
                figure=go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdPu',
                    zmin=-1,
                    zmax=1
                )).update_layout(
                    title='Correlation Matrix',
                    xaxis_title='Features',
                    yaxis_title='Features',
                    width=800,
                    height=800
                )
            ),
        ])
    
    elif tab == 'tab-2':
        return html.Div([
            dcc.Dropdown(
                id='feature-dropdown',
                options=feature_options,
                value=data.columns[2],  # Default value for the dropdown
                style={'width': '50%'},
                clearable=False  # This disables the 'x' button
            ),
            dcc.Graph(id='feature-plot'),
            dcc.Graph(id='echo-histogram'),
            dcc.Graph(id='box-plot'),
            dcc.Graph(id='violin-plot'),
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H2("Logistic Regression Model Performance"),
            html.P(model_performance[0]),
            html.Pre(model_performance[1]),
            html.Pre(model_performance[2]),
            html.Hr(),
            html.H3("Predict Malignancy from Radius"),
            dcc.Input(id='radius-input', type='number', placeholder='Enter radius value'),
            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-output', style={'margin-top': '20px'})
        ])

# Callback for Tab 2 (Feature Analysis) to update plots based on selected feature
@app.callback(
    [Output('feature-plot', 'figure'),
     Output('echo-histogram', 'figure'),
     Output('box-plot', 'figure'),
     Output('violin-plot', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_plots(selected_feature):
    mean_val = feature_stats[selected_feature]['mean']
    std_err_val = feature_stats[selected_feature]['std_err']

    fig1 = px.histogram(data, x=selected_feature, color='diagnosis', barmode='overlay', nbins=100,
                         category_orders={'diagnosis': [0, 1]}, color_discrete_map={0: 'blue', 1: 'red'})
    fig1.update_layout(title=f"Distribution of {selected_feature} and Correlation with Diagnosis",
                       xaxis_title=selected_feature,
                       yaxis_title="Count",
                       showlegend=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=data[selected_feature], nbinsx=50, histnorm='probability', name='Distribution'))
    fig2.add_trace(go.Scatter(x=[mean_val, mean_val], y=[0, 0.15], mode='lines', name='Mean',
                              line=dict(color='red', dash='dash')))
    fig2.add_trace(go.Scatter(x=[mean_val - std_err_val, mean_val + std_err_val], y=[0.05, 0.05], mode='lines',
                              fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)', name='Standard Error'))
    fig2.update_layout(title=f"{selected_feature} Distribution with Mean, Variance, and Standard Error",
                       xaxis_title=selected_feature,
                       yaxis_title="Probability",
                       showlegend=True)

    fig3 = px.box(data, x='diagnosis', y=selected_feature, color='diagnosis', points='all')
    fig3.update_layout(title=f"Box Plot of {selected_feature} by Diagnosis",
                       xaxis_title='Diagnosis',
                       yaxis_title=selected_feature,
                       showlegend=False)

    fig4 = px.violin(data, x='diagnosis', y=selected_feature, color='diagnosis', box=True, points="all",
                     hover_data=data.columns, category_orders={'diagnosis': ['M', 'B']},
                     color_discrete_map={'M': 'red', 'B': 'blue'})
    fig4.update_layout(title=f"Violin Plot of {selected_feature} by Diagnosis",
                       xaxis_title='Diagnosis',
                       yaxis_title=selected_feature,
                       showlegend=False)

    return fig1, fig2, fig3, fig4

# callback for tab-3 logistic regression prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('radius-input', 'value')]
)
def predict_malignancy(n_clicks, radius_value):
    if n_clicks > 0 and radius_value is not None:
        # Create a DataFrame with all the required feature columns
        input_data = pd.DataFrame(columns=X.columns)

        # Set the user-provided radius_mean value
        input_data['radius_mean'] = [radius_value]

        # Fill other features with the mean values (or other default values)
        for feature in input_data.columns:
            if feature != 'radius_mean':  # Already set 'radius_mean' from user input
                input_data[feature] = data_copy[feature].mean()  # Use the mean of the feature from the training data
        
        # Impute any missing values
        input_data_imputed = imputer.transform(input_data)
        
        # Standardize the input data
        input_data_standardized = scaler.transform(input_data_imputed)
        
        # Make the prediction
        prediction = model.predict(input_data_standardized)
        prediction_label = 'Malignant' if prediction[0] == 1 else 'Benign'
        
        return f"The predicted result is: {prediction_label}"
    
    return "Please input a radius value and click predict."


if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
