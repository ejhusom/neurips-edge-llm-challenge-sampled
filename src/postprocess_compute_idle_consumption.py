import argparse
import pandas as pd

def compute_idle_consumption(df: pd.DataFrame, power_column: str = "power", remove_n_first_lines=100, latex: bool = False) -> str:
    """Compute idle consumption from a dataframe

    Args:
        df (pd.DataFrame): Dataframe containing the power consumption
        power_column (str, optional): Column containing the power consumption. Defaults to "power".
        remove_n_first_lines (int, optional): Number of first lines to remove. Defaults to 100.
        latex (bool, optional): Format the report as a LaTeX table. Defaults to False.

    Returns:
        str: Idle consumption report
    """

    # Remove n first lines
    df = df[remove_n_first_lines:]

    # # Compute statistics (mean, std, min, max)
    # mean_power = df[power_column].mean()
    # std_power = df[power_column].std()
    # min_power = df[power_column].min()
    # max_power = df[power_column].max()


    # Computer statistics and put it in a dataframe
    statistics = df[power_column].describe()
    statistics = statistics.to_frame().T

    # Create a report, either as a string or as a LaTeX table
    if latex:
        report = statistics.to_latex(
            float_format="%.2f",
            caption="Idle consumption",
            label="tab:idle_consumption"
        )
    else:
        report = statistics.to_string()

    return report

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute idle consumption")
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("output_file", help="Output file without extension", default="idle_consumption_report")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    results = compute_idle_consumption(df)

    with open(args.output_file + ".txt", "w") as f:
        f.write(results)

    results = compute_idle_consumption(df, latex=True)

    with open(args.output_file + ".tex", "w") as f:
        f.write(results)