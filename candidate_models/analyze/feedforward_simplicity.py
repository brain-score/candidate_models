import os

import pandas as pd


def main():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'feedforward_simplicity.tsv'), delim_whitespace=True)
    data = data.sort_values('Feedforward-Simplicity', ascending=False)
    print(data[['model', 'Feedforward-Simplicity', 'operations']])


if __name__ == '__main__':
    main()
