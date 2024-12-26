import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
in_vehicle_coupon_recommendation = fetch_ucirepo(id=603) 
  
# # data (as pandas dataframes) 
# X = in_vehicle_coupon_recommendation.data.features 
# y = in_vehicle_coupon_recommendation.data.targets 
  
# metadata 
in_vehicle_coupon_recommendation.metadata
  
# variable information 
in_vehicle_coupon_recommendation.variables

# dataframe = pd.concat([X, y])
# dataframe



# Title of the Streamlit app
st.title("Data Exploration & Visualization")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)

    # Initialize session state for dataframe if not already done
    if 'df' not in st.session_state:
        st.session_state.df = df

    # Initialize session state for column data types if not already done
    if 'column_dtype_dict' not in st.session_state:
        st.session_state.column_dtype_dict = {col: str(df[col].dtype) for col in df.columns}

    # Sidebar for selecting number of rows to display
    st.sidebar.header("Display Settings")
    num_rows = st.sidebar.slider("Select number of rows to display", min_value=1, max_value=len(st.session_state.df), value=5, step=1)

    # Display the DataFrame
    st.write(f"First {num_rows} rows of the dataframe:")
    st.write(st.session_state.df.head(num_rows))

    # Create a dictionary with column index and column names
    index_column_dict = {str(i): col for i, col in enumerate(st.session_state.df.columns)}

    # # Sidebar for setting data types
    # st.sidebar.header("Set Data Types")

    # # Allow user to select the column for setting data type
    # column_to_set_dtype = st.sidebar.selectbox("Select column to set data type", options=list(st.session_state.df.columns))

    # # Allow user to select the datatype for the chosen column
    # data_types = ['int64', 'float64', 'string', 'datetime64[ns]', 'bool']
    # selected_dtype = st.sidebar.selectbox(f"Select data type for {column_to_set_dtype}", options=data_types)

    # # Apply the selected data type to the chosen column
    # try:
    #     st.session_state.df[column_to_set_dtype] = st.session_state.df[column_to_set_dtype].astype(selected_dtype)
    #     st.session_state.column_dtype_dict[column_to_set_dtype] = selected_dtype
    # except ValueError as e:
    #     st.sidebar.error(f"Error converting column {column_to_set_dtype} to {selected_dtype}: {e}")

    # Check for duplicate values
    duplicate = df[df.duplicated(keep = "last")]
    st.write("Duplicate records in the dataset:")
    duplicate

    # removing duplicate records from the dataset
    st.write("Dropping duplicate values from the dataset...")
    df = df.drop_duplicates()
    st.write("Shape of the dataset after removing duplicates:")
    st.write(df.shape)

    # Missing Values
    # Create a dictionary with column names and missing value count
    missing_values_dict = {col: int(st.session_state.df[col].isnull().sum()) for col in st.session_state.df.columns}

    # Create columns for side by side display
    col1, col2, col3 = st.columns(3)

    # Display the dictionaries side by side
    with col1:
        st.write("Column Names:")
        st.write(index_column_dict)
    with col2:
        st.write("Column Data Types:")
        st.write(st.session_state.column_dtype_dict)
    with col3:
        st.write("Missing Values:")
        st.write(missing_values_dict)


    # Sidebar for missing value handling
    st.sidebar.header("Missing Value Handling")

    # User input for column drop threshold
    col_threshold_percent = st.sidebar.slider("Select column missing values threshold (%)", min_value=0, max_value=100, value=25, step=1)
    col_threshold = len(df) * (col_threshold_percent / 100)

    # Identify and drop columns with missing values above threshold
    cols_to_drop = df.columns[df.isnull().mean() * 100 > col_threshold_percent].tolist()
    df = df.drop(cols_to_drop, axis=1)

    # Display dropped columns
    st.write("Dropped Columns:")
    if cols_to_drop:
        st.write(cols_to_drop)
    else:
        st.write("No columns were dropped.")

    dropped_rows = df[df.isnull().any(axis=1)]
    df = df.dropna()

    # Display only dropped rows
    st.write("Dropped Rows:")
    if not dropped_rows.empty:
        st.write(dropped_rows)
    else:
        st.write("No rows were dropped.")

    
    # Display number of missing values after handling with column names in a row format
    missing_values_after_handling_dict = {col: int(df[col].isnull().sum()) for col in df.columns}
    st.write("Number of missing values after handling:")
    st.write(', '.join([f"{col}: {missing_values_after_handling_dict[col]}" for col in missing_values_after_handling_dict]))
    
    # Display the DataFrame after handling missing values
    st.write("Shape of the dataframe after handling missing values:")
    st.write(df.shape)

    # Sidebar for data subsetting
    st.sidebar.header("Data Subsetting")
    
    st.write("Subsetting Dataframe:")
    # Allow user to select the column for subsetting
    column_filter = st.sidebar.selectbox("Select column to filter by", options=['All'] + list(df.columns))
    
    # Initialize subset_df with the full DataFrame
    subset_df = df
    
    if column_filter != 'All':
        # Allow user to select the value for the chosen column
        unique_values = df[column_filter].unique()
        value_filter = st.sidebar.selectbox(f"Select value for {column_filter}", options=['All'] + list(unique_values))
        
        # Subset the DataFrame based on user selection
        if value_filter != 'All':
            subset_df = df[df[column_filter] == value_filter]
    
    # Display the subset DataFrame
    st.write(f"First {num_rows} rows of the dataframe subset{'' if column_filter == 'All' else f' for {column_filter} = {value_filter}'}: ")
    st.write(subset_df.head(num_rows))

    
    # Basic Statistics
    st.write("Basic Statistics:")
    st.write(subset_df.describe(include='all'))

    st.write("Correlation Heatmap:")
    numerical_df = subset_df.select_dtypes(include=["float64", "int64"])

    if not numerical_df.empty:
        corr_matrix = np.round(numerical_df.corr(), 2)
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        color_continuous_scale='RdBu')
        st.plotly_chart(fig)
    else:
        st.write("No numerical columns available for correlation heatmap.")


    # Sidebar for selecting columns and chart types
    st.sidebar.header("Visualization Settings")
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Bar Chart"])

    # Separate columns by data type
    numerical_columns = subset_df.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = subset_df.select_dtypes(include=["object", "string"]).columns
    all_columns = subset_df.columns

    # Plot Histogram
    if chart_type == "Histogram":
        if numerical_columns.empty:
            st.sidebar.write("No numerical columns available for histogram.")
        else:
            column = st.sidebar.selectbox("Select Column for Histogram", numerical_columns)
            bins = st.sidebar.slider("Number of Bins", 5, 50, 20)

            # Let the user pick a column to color the histogram
            color_column = st.sidebar.selectbox("Select Column for Coloring", [None] + list(categorical_columns))

            # Plot histogram using plotly
            if color_column:
                # Compute the order of categories
                ordered_categories = subset_df[color_column].value_counts().sort_values(ascending=True).index.tolist()

                figure = px.histogram(
                    subset_df,
                    x=column,
                    nbins=bins,
                    color=color_column,  # Use the selected column for coloring
                    title=f'{column} Distribution',
                    labels={column: column, 'count': 'Count'},
                    category_orders={color_column: ordered_categories}  # Sort categories
                )
            else:
                # Plot without color if no color column is selected
                figure = px.histogram(
                    subset_df,
                    x=column,
                    nbins=bins,
                    title=f'{column} Distribution',
                    labels={column: column, 'count': 'Count'}
                )

            # Display the plotly figure
            st.plotly_chart(figure)

    elif chart_type == "Bar Chart":
        if categorical_columns.empty:
            st.sidebar.write("No categorical columns available for bar chart.")
        else:
            column = st.sidebar.selectbox("Select Column for Bar Chart", list(all_columns))

            # Let the user pick a column to color the bar chart
            color_column = st.sidebar.selectbox("Select Column for Coloring", [None] + list(all_columns))

            # Let the user select whether to display percentages or counts
            display_mode = st.sidebar.radio("Display Mode", ["Count", "Percentage"])

            if display_mode == "Count":
                # Group by the selected column and count the occurrences
                if color_column:
                    bar_data = subset_df.groupby([column, color_column]).size().reset_index(name='count')
                    bar_data.columns = [column, color_column, 'count']
                else:
                    bar_data = subset_df[column].value_counts().reset_index()
                    bar_data.columns = [column, 'count']
                y_value = 'count'
                y_title = 'Count'
            else:
                total_counts = subset_df[column].value_counts()
                percentages = (total_counts / total_counts.sum()) * 100
                percent_df = percentages.reset_index().rename(columns={'index': column, column: 'Percentage'})
                if color_column:
                    color_total_counts = subset_df.groupby([column, color_column]).size().reset_index(name='count')
                    color_total_counts['Percentage'] = (color_total_counts['count'] / color_total_counts.groupby(column)['count'].transform('sum')) * 100
                    bar_data = color_total_counts
                else:
                    bar_data = percent_df
                y_value = 'Percentage'
                y_title = 'Percentage (%)'

            # Plot bar chart using plotly
            figure = px.bar(
                bar_data,
                x=column,
                y=y_value,
                color=color_column if color_column else None,
                title=f'{column} {y_title}',
                labels={column: column, y_value: y_title}
            )

            # Make sure all x-axis categories are displayed
            figure.update_layout(xaxis=dict(type='category'))
            
            # Display the plotly figure
            st.plotly_chart(figure)



    # Sidebar for selecting the column and conditions for coupon acceptance analysis
    st.sidebar.header("Coupon Acceptance Analysis")

    target_column = st.sidebar.selectbox("Select column for coupon acceptance analysis", options=list(subset_df.columns))

    selected_columns = st.sidebar.multiselect("Select columns for conditions", options=list(subset_df.columns))

    conditions = {}

    for col in selected_columns:
        unique_values = subset_df[col].unique()
        selected_values = st.sidebar.multiselect(f"Select values for {col}", options=unique_values, key=col)
        conditions[col] = selected_values

    if target_column:
        filtered_df = subset_df.copy()
        for col, values in conditions.items():
            if values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        if not filtered_df.empty:
            accepted_coupon = (filtered_df[target_column] == 1).mean()
            rejected_coupon = 1 - accepted_coupon
            st.write(f"The acceptance rate of coupon is {accepted_coupon:.2f}")
            st.write(f"The rejection rate of coupon is {rejected_coupon:.2f}")
        else:
            st.write("No data available for the selected conditions.")
    else:
        st.write("Please select a column for coupon acceptance analysis.")


else:
    st.write("Please upload a CSV file to get started.")





# import streamlit as st
# import numpy as np

# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)



# import streamlit as st
# import numpy as np
# import pandas as pd

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))

# st.dataframe(dataframe.style.highlight_max(axis=0))


# import streamlit as st
# import numpy as np
# import pandas as pd

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.table(dataframe)

# import streamlit as st
# import numpy as np
# import pandas as pd

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# import streamlit as st
# import numpy as np
# import pandas as pd

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

# import streamlit as st
# x = st.slider('x')  # 👈 this is a widget, treat widgets as variables
# st.write(x, 'squared is', x * x)



# import streamlit as st
# st.text_input("Your name", key="name")

# # You can access the value at any point with:
# st.session_state.name


# import streamlit as st
# import numpy as np
# import pandas as pd

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#        np.random.randn(20, 3),
#        columns=['a', 'b', 'c'])

#     chart_data


# # df = pd.DataFrame({
# #     'first column': [1, 2, 3, 4],
# #     'second column': [10, 20, 30, 40]
# #     })

# # option = st.selectbox(
# #     'Which number do you like best?',
# #      df['first column'])

# # 'You selected: ', option


# import streamlit as st

# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )


# import streamlit as st

# left_column, right_column = st.columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")


# import streamlit as st
# import time

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'







    # coffee_house_df = df[df['coupon'] == 'Coffee House']

    # color_map = {0: 'blue', 1: 'green'}
    # coffee_house_df['color'] = coffee_house_df['Y'].map(color_map)
    # df = coffee_house_df[['passanger', 'weather', 'age', 'expiration', 'has_children', 'direction_same', 'direction_opp', 'Y', 'color']]

    # fig = px.parallel_categories(df,
    #                             dimensions=['age', 'passanger', 'weather', 'expiration', 'has_children', 'direction_same', 'direction_opp', 'Y'],
    #                             color='color',
    #                             labels={'Y': 'Coupon_Accepted'},
    #                             color_continuous_midpoint=0.5
    #                             )

    # fig