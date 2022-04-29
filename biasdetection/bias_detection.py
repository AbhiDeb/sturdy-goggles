import sys

sys.path.append('..')
import pandas as pd
import plotly.graph_objects as go

from modeldevelopment import settings

target_column = settings.Y_COLUMN
fav_class = settings.FAVORABLE_CLASS[0]
un_related_columns = settings.COlUMN_TO_DROP

data = pd.read_csv(settings.DATASET_PATH, na_values=["Unknown", " "])
data_orig = data.copy()

ignored_columns = un_related_columns

# Drop the all ignored columns.
data.drop(columns=ignored_columns, inplace=True)
# filter the category and numerical columns
cat_columns = []
for col in data.columns:
    if data[col].nunique() <= 15 and col not in target_column:
        cat_columns.append(col)

num_columns = []
for col in data.columns:
    if col not in cat_columns:
        num_columns.append(col)

# set the imputed value for all features
filling_values = dict()
for each_column in num_columns + cat_columns:
    if data[each_column].dtypes in ["int32", "int64", "float32", "float64"]:
        filling_values[each_column] = data[each_column].median()
    elif data[each_column].dtypes in ["object"]:
        filling_values[each_column] = "Missing value"

data.fillna(filling_values, inplace=True)

column_values = []
priv_values = []
unpriv_values = []
priv_pct_values = []
unpriv_pct_values = []
spd_values = []
num_cat_columns = dict()
for each_column in cat_columns:

    cat_sorted = dict()
    sub_cat = set(data[each_column])

    for each_sub_cat in sub_cat:
        cat_sorted[str(each_sub_cat)] = len(
            data[
                (data[each_column] == each_sub_cat)
                & (data[target_column[0]] == fav_class)
                ]
        ) / len(data[(data[each_column] == each_sub_cat)])

    cat_sorted_sorted = dict(
        sorted(cat_sorted.items(), key=lambda x: x[1], reverse=True)
    )
    num_cat_columns[str(each_column)] = list(cat_sorted_sorted.keys())

    iter_items = iter(cat_sorted_sorted.items())
    top_item = next(iter_items)
    priv_cat, priv_pct = top_item[0], top_item[1]

    for cat, pct in iter_items:
        unpriv_cat, unpriv_pct = cat, pct

        spd_values.append(round(unpriv_pct - priv_pct, 5))
        column_values.append(each_column)
        priv_values.append(priv_cat)
        priv_pct_values.append(round(priv_pct * 100, 2))
        unpriv_pct_values.append(round(unpriv_pct * 100, 2))
        unpriv_values.append(unpriv_cat)

column_values = []
priv_values = []
unpriv_values = []
priv_pct_values = []
unpriv_pct_values = []
spd_values = []
num_cat_columns = dict()
for each_column in cat_columns:

    cat_sorted = dict()
    sub_cat = set(data[each_column])

    for each_sub_cat in sub_cat:
        cat_sorted[str(each_sub_cat)] = len(
            data[
                (data[each_column] == each_sub_cat)
                & (data[target_column[0]] == fav_class)
                ]
        ) / len(data[(data[each_column] == each_sub_cat)])

    cat_sorted_sorted = dict(
        sorted(cat_sorted.items(), key=lambda x: x[1], reverse=True)
    )
    num_cat_columns[str(each_column)] = list(cat_sorted_sorted.keys())

    iter_items = iter(cat_sorted_sorted.items())
    top_item = next(iter_items)
    priv_cat, priv_pct = top_item[0], top_item[1]

    for cat, pct in iter_items:
        unpriv_cat, unpriv_pct = cat, pct

        spd_values.append(round(unpriv_pct - priv_pct, 5))
        column_values.append(each_column)
        priv_values.append(priv_cat)
        priv_pct_values.append(round(priv_pct * 100, 2))
        unpriv_pct_values.append(round(unpriv_pct * 100, 2))
        unpriv_values.append(unpriv_cat)

df = pd.DataFrame()

df["Feature"] = column_values
df["Priviliged"] = priv_values
df["UnPriviliged"] = unpriv_values
df["% Priviliged class"] = priv_pct_values
df["SPD"] = spd_values
df.to_csv('./reports/BiasDetection.csv', index=False)


def draw_table(column_names, column_values, title):
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=column_names,
                    line_color="darkslategray",
                    fill_color="lightskyblue",
                    align="left",
                ),
                cells=dict(
                    values=column_values,
                    line_color="darkslategray",
                    fill_color="lightcyan",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        autosize=True,
        title={
            "text": f"Bias-Detection for {title.strip()} Categorical feature",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )
    fig.write_image(f"./reports/figures/{title.strip()}.png")


binary_cat = dict(filter(lambda elem: len(elem[1]) == 2, num_cat_columns.items()))

binary_df = df[df["Feature"].isin(binary_cat.keys())].reset_index(drop=True)
draw_table(
    binary_df.columns,
    [binary_df[cat] for cat in binary_df.columns],
    "Binary",
)

multi_cat = [each_col for each_col in cat_columns if each_col not in binary_cat]

multi_df = df[df["Feature"].isin(multi_cat)].reset_index(drop=True)

for each_col in multi_cat:

    def_columns = ["Feature", "Privileged Class", "% Priviliged class"]
    col_values = []

    temp_df = multi_df[multi_df["Feature"] == str(each_col)]

    col_values.append([each_col])
    col_values.append([temp_df["Priviliged"].iloc[0]])
    col_values.append([temp_df["% Priviliged class"].iloc[0]])

    for index, row in temp_df.iterrows():
        un_priv = f"SPD with \n{row['UnPriviliged']}"
        spd = row["SPD"]
        def_columns.append([un_priv])
        col_values.append([spd])

    draw_table(column_names=def_columns,
               column_values=col_values,
               title=each_col)
