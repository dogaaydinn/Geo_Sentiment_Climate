"""
Missing Value Comparison Module.

Provides comprehensive comparison and analysis of missing values
between original and processed datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from source.utils.logger import setup_logger
from source.config.config_utils import config

logger = setup_logger(
    name="missing_value_comparison",
    log_file="../logs/missing_value_comparison.log",
    log_level="INFO"
)


def calculate_missing_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive missing value statistics for a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing value statistics
    """
    stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values,
        'total_count': len(df),
        'non_missing_count': df.notna().sum().values
    })

    stats = stats.sort_values('missing_percentage', ascending=False)
    stats = stats.reset_index(drop=True)

    return stats


def compare_missing_values(
    df_original: pd.DataFrame,
    df_processed: pd.DataFrame,
    save_report: bool = True,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Compare missing values between original and processed DataFrames.

    This function is called from data_preprocessing.py to analyze
    the effectiveness of missing value imputation.

    Args:
        df_original: Original DataFrame before processing
        df_processed: Processed DataFrame after imputation
        save_report: Whether to save a comparison report
        save_plots: Whether to save comparison plots

    Returns:
        Dictionary containing comparison statistics
    """
    logger.info("=" * 80)
    logger.info("Missing Value Comparison Analysis")
    logger.info("=" * 80)

    # Calculate statistics for both datasets
    stats_original = calculate_missing_statistics(df_original)
    stats_processed = calculate_missing_statistics(df_processed)

    # Merge statistics
    comparison = pd.merge(
        stats_original[['column', 'missing_count', 'missing_percentage']],
        stats_processed[['column', 'missing_count', 'missing_percentage']],
        on='column',
        suffixes=('_original', '_processed')
    )

    # Calculate improvement
    comparison['missing_reduced'] = (
        comparison['missing_count_original'] - comparison['missing_count_processed']
    )
    comparison['percentage_improved'] = (
        comparison['missing_percentage_original'] - comparison['missing_percentage_processed']
    )
    comparison['reduction_rate'] = (
        comparison['missing_reduced'] / comparison['missing_count_original'].replace(0, np.nan) * 100
    )

    # Sort by improvement
    comparison = comparison.sort_values('missing_reduced', ascending=False)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics:")
    logger.info("=" * 80)

    total_missing_original = stats_original['missing_count'].sum()
    total_missing_processed = stats_processed['missing_count'].sum()
    total_cells = len(df_original) * len(df_original.columns)

    logger.info(f"Total cells: {total_cells:,}")
    logger.info(f"Missing values (original): {total_missing_original:,} "
                f"({total_missing_original/total_cells*100:.2f}%)")
    logger.info(f"Missing values (processed): {total_missing_processed:,} "
                f"({total_missing_processed/total_cells*100:.2f}%)")
    logger.info(f"Values imputed: {total_missing_original - total_missing_processed:,}")
    logger.info(f"Reduction rate: {(1 - total_missing_processed/total_missing_original)*100:.2f}%")

    # Log top improvements
    logger.info("\n" + "=" * 80)
    logger.info("Top 10 Columns by Missing Value Reduction:")
    logger.info("=" * 80)

    top_improvements = comparison.nlargest(10, 'missing_reduced')
    for idx, row in top_improvements.iterrows():
        logger.info(
            f"{row['column']:40s}: "
            f"{int(row['missing_count_original']):>6d} -> {int(row['missing_count_processed']):>6d} "
            f"({row['percentage_improved']:>6.2f}% improved)"
        )

    # Identify columns still with missing values
    still_missing = comparison[comparison['missing_count_processed'] > 0]
    if not still_missing.empty:
        logger.info("\n" + "=" * 80)
        logger.info(f"Columns Still With Missing Values: {len(still_missing)}")
        logger.info("=" * 80)
        for idx, row in still_missing.iterrows():
            logger.info(
                f"{row['column']:40s}: "
                f"{int(row['missing_count_processed']):>6d} "
                f"({row['missing_percentage_processed']:>6.2f}%)"
            )

    # Create visualizations
    if save_plots:
        plots_dir = Path(config.get("paths", {}).get("plots_dir", "../plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Before/After comparison
        plot_before_after_comparison(
            comparison,
            save_path=plots_dir / "missing_values_before_after.png"
        )

        # Plot 2: Reduction heatmap
        plot_reduction_heatmap(
            comparison,
            save_path=plots_dir / "missing_values_reduction.png"
        )

        # Plot 3: Missing value pattern
        plot_missing_pattern(
            df_original,
            df_processed,
            save_path=plots_dir / "missing_values_pattern.png"
        )

    # Save report
    if save_report:
        reports_dir = Path("../reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "missing_value_comparison_report.csv"
        comparison.to_csv(report_path, index=False)
        logger.info(f"\nComparison report saved to: {report_path}")

    # Prepare return dictionary
    result = {
        'total_missing_original': int(total_missing_original),
        'total_missing_processed': int(total_missing_processed),
        'total_imputed': int(total_missing_original - total_missing_processed),
        'reduction_rate': float((1 - total_missing_processed/total_missing_original) * 100)
        if total_missing_original > 0 else 0.0,
        'columns_fully_imputed': int((comparison['missing_count_processed'] == 0).sum()),
        'columns_still_missing': int((comparison['missing_count_processed'] > 0).sum()),
        'comparison_table': comparison
    }

    logger.info("\n" + "=" * 80)
    logger.info("Missing Value Comparison Analysis Completed")
    logger.info("=" * 80)

    return result


def plot_before_after_comparison(
    comparison: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot before/after comparison of missing values.

    Args:
        comparison: Comparison DataFrame
        save_path: Path to save the plot
    """
    # Get top 20 columns with most missing values originally
    top_cols = comparison.nlargest(20, 'missing_count_original')

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_cols))
    width = 0.35

    bars1 = ax.bar(
        x - width/2,
        top_cols['missing_percentage_original'],
        width,
        label='Original',
        color='indianred',
        alpha=0.8
    )

    bars2 = ax.bar(
        x + width/2,
        top_cols['missing_percentage_processed'],
        width,
        label='Processed',
        color='seagreen',
        alpha=0.8
    )

    ax.set_xlabel('Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Missing Values Comparison: Original vs Processed',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(top_cols['column'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Before/After comparison plot saved to: {save_path}")

    plt.close()


def plot_reduction_heatmap(
    comparison: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot heatmap showing reduction in missing values.

    Args:
        comparison: Comparison DataFrame
        save_path: Path to save the plot
    """
    # Get top 20 columns with most reduction
    top_cols = comparison.nlargest(20, 'missing_reduced')

    # Prepare data for heatmap
    heatmap_data = top_cols[['missing_percentage_original', 'missing_percentage_processed']].T
    heatmap_data.columns = top_cols['column']
    heatmap_data.index = ['Original', 'Processed']

    fig, ax = plt.subplots(figsize=(16, 4))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Missing Percentage (%)'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title(
        'Missing Values Heatmap: Top 20 Improved Columns',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reduction heatmap saved to: {save_path}")

    plt.close()


def plot_missing_pattern(
    df_original: pd.DataFrame,
    df_processed: pd.DataFrame,
    save_path: Optional[Path] = None,
    max_cols: int = 30
) -> None:
    """
    Plot missing value patterns for original and processed data.

    Args:
        df_original: Original DataFrame
        df_processed: Processed DataFrame
        save_path: Path to save the plot
        max_cols: Maximum number of columns to display
    """
    # Select columns with any missing values in either dataset
    cols_with_missing = []
    for col in df_original.columns:
        if df_original[col].isnull().any() or df_processed[col].isnull().any():
            cols_with_missing.append(col)

    # Limit to max_cols
    cols_to_plot = cols_with_missing[:max_cols]

    if not cols_to_plot:
        logger.info("No missing values to plot")
        return

    # Create missing value matrices
    missing_original = df_original[cols_to_plot].isnull().astype(int)
    missing_processed = df_processed[cols_to_plot].isnull().astype(int)

    # Sample rows if too many
    if len(missing_original) > 1000:
        sample_idx = np.random.choice(len(missing_original), 1000, replace=False)
        missing_original = missing_original.iloc[sample_idx]
        missing_processed = missing_processed.iloc[sample_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Original missing pattern
    sns.heatmap(
        missing_original.T,
        cmap='RdYlGn_r',
        cbar=False,
        ax=ax1,
        yticklabels=True,
        xticklabels=False
    )
    ax1.set_title('Missing Value Pattern - Original', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Samples', fontsize=10)
    ax1.set_ylabel('Columns', fontsize=10)

    # Processed missing pattern
    sns.heatmap(
        missing_processed.T,
        cmap='RdYlGn_r',
        cbar=False,
        ax=ax2,
        yticklabels=True,
        xticklabels=False
    )
    ax2.set_title('Missing Value Pattern - Processed', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Samples', fontsize=10)
    ax2.set_ylabel('Columns', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Missing pattern plot saved to: {save_path}")

    plt.close()


def generate_missing_value_report(
    df: pd.DataFrame,
    report_name: str = "missing_value_report"
) -> pd.DataFrame:
    """
    Generate a detailed missing value report.

    Args:
        df: Input DataFrame
        report_name: Name for the report

    Returns:
        DataFrame with detailed missing value statistics
    """
    logger.info(f"Generating missing value report: {report_name}")

    stats = calculate_missing_statistics(df)

    # Add additional statistics
    stats['first_missing_index'] = df[stats['column']].apply(
        lambda col: df[col].isnull().idxmax() if df[col].isnull().any() else None
    ).values

    stats['last_missing_index'] = df[stats['column']].apply(
        lambda col: df[col][::-1].isnull().idxmax() if df[col].isnull().any() else None
    ).values

    # Save report
    reports_dir = Path("../reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{report_name}.csv"
    stats.to_csv(report_path, index=False)

    logger.info(f"Report saved to: {report_path}")

    return stats
