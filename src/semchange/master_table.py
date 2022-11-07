"""Script to generate the new master table fromn ARC output.
"""

import pandas as pd
import numpy as np
import glob
import os
import click

@click.command()
@click.option('--scan_dir', required=True, help='directory within which to scan')
@click.option('--')
def main():
    pass

if __name__ == '__main__':
    main()