import pandas as pd
from itertools import groupby
from master import MapReduceMaster
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

FILENAME = "COVID-19_US_County-level_Summaries-master\\COVID-19_US_County-level_Summaries-master\\data\\infections_timeseries.csv"
N_MONTHS = 3
TOP_K = 5

def user_mapper(record):
    """
    Input: Dataframe Series with county information.
    Output: Tuples list ((State, Period), (County, TotalCases))
    """

    try:
        date_cols = [c for c in record.index if '/' in str(c)]
        output = list(
            map(
                lambda item: (
                    # Returns ((state, time period), (county, total of new cases))
                    (record.Combined_Key.split('-')[-2].strip(),
                    f"({item[0].strftime('%b')} - {(item[0] + pd.DateOffset(months=N_MONTHS-1)).strftime('%b')}) {item[0].strftime('%Y')}"),
                    (record.Combined_Key.split('-')[0].strip(), int(item[1]))
                ),
                # Provides (time period, total of new cases)
                pd.Series(
                    map(
                        # Corrects any negative value to zero
                        lambda x: max(0, x), 
                        map(
                            # Instead of cumulative values, we want to find out how many new cases we got on each date
                            lambda curr_date, previous_date: curr_date - previous_date,
                            record[date_cols].tolist(), [0] + record[date_cols].tolist()
                        )
                    ),
                    # Arrange the map result to be able to group it by month frequency
                    index=pd.to_datetime(date_cols, format='%m/%d/%y')
                ).groupby(pd.Grouper(freq=f"{N_MONTHS}MS")).sum()#, origin=pd.Timestamp('2020-01-01'))).sum()
                .items()
            ),
        )
        return output
    except Exception as e:
        return []

def user_shuffler(mapped_data, num_workers):
    buckets = {i: [] for i in range(num_workers)}
    for key, value in mapped_data:
        # The name of the state will be the decisive factor to which worker it should go
        target = sum(bytearray(key[0].encode('utf-8'))) % num_workers
        buckets[target].append((key, value))
    return buckets

def user_reducer(received_data):
    """
    Input: ((State, Period),(County, TotalCases))
    Output: (State, Period, Top_K_List)
    """

    output = list(
        map(
            lambda item:
                (item["key"][0], item["key"][1], (lambda target_values: 
                        list(filter(
                            lambda c: c[1] in target_values, 
                            item["counties"]
                        ))
                    )(
                        # Returns the TOP_K values of most new cases in the period
                        set(sorted(list(set(map(lambda x: x[1], item["counties"]))), reverse=True)[:TOP_K])
                    )
                ),
            # Returns {key = (state, period), counties = list of (county, totalcases)}
            list(
                map(
                    lambda item: {"key": item[0], "counties": list(map(lambda value: value[1], item[1]))},
                    groupby(sorted(received_data, key=lambda x: x[0]), key=lambda x: x[0])
                )
            )
        )
    )

    return output

def parse_period(p_str):
            # p_str: "(Jan - Mar) 2020" -> ['Jan', '-', 'Mar', '2020']
            parts = p_str.replace("(", "").replace(")", "").split()
            return pd.to_datetime(f"{parts[0]} {parts[3]}", format="%b %Y")

def user_result_handler(results, worker_id):
    """
    Receives the Reducer results and generates a horizontal bar chart
    for each State present in the data of this Worker.
    Adds data labels (case counts) to each bar.
    """

    if not results:
        print(f"[Worker {worker_id}] No results to generate charts.")
        return

    print(f"[Worker {worker_id}] Starting chart generation...")

    # 1. Flatten the data into a tabular format
    # From: [(State, Period, [(County, Cases), ...]), ...]
    # To: A simple list of dictionaries
    flat_data = []
    for state, period, top_k_list in results:
        for county, cases in top_k_list:
            flat_data.append({
                'State': state,
                'Period': period,
                'County': county,
                'Cases': cases
            })
    
    df = pd.DataFrame(flat_data)

    sns.set_theme(style="whitegrid")
    
    # 2. Create folder
    output_dir = f"charts_worker_{worker_id}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Iterate through each State
    unique_states = df['State'].unique()
    
    for state in unique_states:
        state_df = df[df['State'] == state]
        periods = state_df['Period'].unique()
        
        chronological_order = sorted(periods, key=parse_period)

        plt.figure(figsize=(14, 8))
        
        ax = sns.barplot(
            data=state_df,
            y="Period",    # Vertical Axis: Periods
            x="Cases",     # Horizontal Axis: Total Cases
            hue="County",  # Legend/Colors: Counties
            orient="h",    # Horizontal orientation
            palette="mako",
            order=chronological_order
        )

        # Loops through the bar containers and adds the text label
        _, legend_labels = ax.get_legend_handles_labels()
        for idx, container in enumerate(ax.containers):
            county_name = legend_labels[idx]
            custom_labels = [
                f"({county_name}, {bar.get_width():.0f})" if bar.get_width() > 0 else "" 
                for bar in container
            ]
            
            ax.bar_label(
                container, 
                labels=custom_labels,
                padding=4,   
                fontsize=10,
                color="#333333", 
                fontweight="bold"
            )

        # 4. Customization
        if ax.get_legend():
            ax.get_legend().remove()
        plt.title(f"Top {TOP_K} Counties with Most New Cases - {state}", fontsize=16, pad=20)
        plt.xlabel("Total New Cases in Period", fontsize=12)
        plt.ylabel("Period", fontsize=12)

        margin_factor = 1.3 
        max_val = state_df['Cases'].max()
        ax.set_xlim(0, max_val * margin_factor)
        
        plt.tight_layout()

        # 6. Save the file
        safe_state_name = "".join([c for c in state if c.isalnum() or c in (' ', '-', '_')]).strip()
        filename = f"{safe_state_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=120) 
        plt.close() 

    print(f"[Worker {worker_id}] Charts successfully generated in folder '{output_dir}/'")


if __name__ == "__main__":
    master = MapReduceMaster(
        filename=FILENAME, 
        num_workers=2,
        mapper=user_mapper,
        shuffler=user_shuffler,
        reducer=user_reducer,
        result_handler=user_result_handler
    )
    master.start()