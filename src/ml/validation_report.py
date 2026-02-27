"""
ML Pattern Discovery Validation Report Generator.

This module generates formatted markdown reports from ValidationResult objects,
providing comprehensive documentation of validation outcomes.

Following TDD methodology:
- Stream D: Report Generation (Issue #292)
"""

from src.ml.validation import ValidationResult


def generate_validation_report(result: ValidationResult) -> str:
    """
    Generate formatted markdown validation report.

    Args:
        result: ValidationResult object with all validation components

    Returns:
        Formatted markdown string report
    """
    # Format timestamp
    timestamp_str = result.timestamp.strftime("%Y-%m-%d")

    # Determine overall status
    if result.all_success:
        overall_status = "VALIDATION PASSED ✓"
        status_symbol = "✓"
    else:
        overall_status = "VALIDATION FAILED ✗"
        status_symbol = "✗"

    # Build report sections
    report_lines = []

    # Header
    report_lines.extend(
        [
            "ML PATTERN DISCOVERY VALIDATION REPORT",
            "═" * 63,
            "",
            f"VALIDATION DATE: {timestamp_str}",
            "SYSTEM: ML Pattern Discovery (Phase 4)",
            "",
        ]
    )

    # Success Criteria Summary
    report_lines.extend(
        [
            "SUCCESS CRITERIA SUMMARY",
            "─" * 24,
        ]
    )

    # Pattern count
    count_status = "✓" if result.pattern_count.success else "✗"
    report_lines.append(
        f"{count_status} Pattern Count:           {result.pattern_count.count} discovered "
        f"(target: {result.pattern_count.min_patterns}-{result.pattern_count.target_patterns}) "
        f"{count_status}"
    )

    # Profitability
    prof_status = "✓" if result.profitability.success else "✗"
    prof_rate_pct = int(result.profitability.profitability_rate * 100)
    min_pct = int(result.profitability.min_percentage * 100)
    report_lines.append(
        f"{prof_status} Profitability:           {prof_rate_pct}% Sharpe > 0.5 "
        f"(target: {min_pct}%+) {prof_status}"
    )

    # Diversity
    div_status = "✓" if result.diversity.success else "✗"
    report_lines.append(
        f"{div_status} Diversity:               {len(result.diversity.families)} families, "
        f"{len(result.diversity.regimes)} regimes (target: 3 each) {div_status}"
    )

    # Out-of-sample
    gen_status = "✓" if result.generalization.success else "✗"
    degradation_pct = result.generalization.avg_degradation * 100
    max_deg_pct = result.generalization.max_degradation * 100
    report_lines.append(
        f"{gen_status} Out-of-Sample:           Degradation {degradation_pct:.1f}% < {max_deg_pct:.0f}% {gen_status}"
    )

    # Persistence
    persist_pct = int(result.generalization.persistence_rate * 100)
    report_lines.append(
        f"{gen_status} Pattern Persistence:     {persist_pct}% maintain Sharpe > 0.0 {gen_status}"
    )

    # ML vs Manual
    comp_status = "✓" if result.comparison.ml_superior else "✗"
    sharpe_imp_pct = result.comparison.sharpe_improvement * 100
    report_lines.append(
        f"{comp_status} ML vs Manual:            {sharpe_imp_pct:+.1f}% Sharpe improvement {comp_status}"
    )

    report_lines.extend(
        [
            "",
            f"OVERALL STATUS: {overall_status}",
            "",
        ]
    )

    # Detailed Results
    report_lines.extend(
        [
            "DETAILED RESULTS",
            "─" * 16,
            "",
        ]
    )

    # 1. Pattern Count Validation
    report_lines.extend(
        [
            "1. PATTERN COUNT VALIDATION",
            f"   Total patterns discovered: {result.pattern_count.count}",
            f"   Minimum required: {result.pattern_count.min_patterns}",
            f"   Target: {result.pattern_count.target_patterns}",
            f"   Status: {status_symbol if result.pattern_count.success else '✗'} "
            f"{'PASS' if result.pattern_count.success else 'FAIL'} "
            f"({'meets minimum' if result.pattern_count.meets_minimum else 'below minimum'})",
            "",
        ]
    )

    # 2. Profitability Validation
    report_lines.extend(
        [
            "2. PROFITABILITY VALIDATION",
            f"   Total patterns: {result.profitability.total_patterns}",
            f"   Profitable (Sharpe > 0.5): {result.profitability.profitable_count} {status_symbol}",
            f"   Profitability rate: {result.profitability.profitability_rate * 100:.0f}%",
            f"   Minimum required: {result.profitability.min_percentage * 100:.0f}%",
            f"   Status: {status_symbol if result.profitability.success else '✗'} "
            f"{'PASS' if result.profitability.success else 'FAIL'} "
            f"({'exceeds minimum' if result.profitability.meets_threshold else 'below minimum'})",
            "",
        ]
    )

    # 3. Diversity Validation
    report_lines.extend(
        [
            "3. DIVERSITY VALIDATION",
            "   Pattern families:",
        ]
    )
    for family in sorted(result.diversity.families):
        report_lines.append(f"   - {family}")
    report_lines.append(
        f"   Families: {len(result.diversity.families)} (target: 3) {status_symbol}"
    )
    report_lines.append("")
    report_lines.append("   Market regimes covered:")
    for regime in sorted(result.diversity.regimes):
        report_lines.append(f"   - {regime}")
    report_lines.append(
        f"   Regimes: {len(result.diversity.regimes)} (target: 3) {status_symbol}"
    )
    report_lines.append("")
    report_lines.append("   Pattern correlation:")
    report_lines.append(
        f"   - Average correlation: {result.diversity.avg_correlation:.2f} (< 0.5 threshold) {status_symbol}"
    )
    report_lines.append("")
    div_status_text = (
        "PASS (excellent diversity)"
        if result.diversity.success
        else "FAIL (insufficient diversity)"
    )
    report_lines.append(
        f"   Status: {status_symbol if result.diversity.success else '✗'} {div_status_text}"
    )
    report_lines.append("")

    # 4. Out-of-Sample Generalization
    report_lines.extend(
        [
            "4. OUT-OF-SAMPLE GENERALIZATION",
            f"   Total patterns: {result.generalization.total_patterns}",
            f"   Tested patterns: {result.generalization.tested_patterns}",
            "",
            f"   Average degradation: {result.generalization.avg_degradation * 100:.1f}% "
            f"(< {result.generalization.max_degradation * 100:.0f}% threshold) {status_symbol}",
            f"   Persistent patterns (Sharpe > 0.0 in test): "
            f"{result.generalization.persistent_patterns}/{result.generalization.total_patterns} "
            f"({result.generalization.persistence_rate * 100:.0f}%)",
            f"   Minimum required: {result.generalization.min_persistence * 100:.0f}% persistence",
            "",
            f"   Status: {status_symbol if result.generalization.success else '✗'} "
            f"{'PASS (excellent persistence)' if result.generalization.success else 'FAIL (poor generalization)'}",
            "",
        ]
    )

    # 5. Comparative Analysis
    report_lines.extend(
        [
            "5. COMPARATIVE ANALYSIS (ML vs Manual)",
            f"   Manual patterns: {result.comparison.manual_count} patterns",
            f"   ML patterns: {result.comparison.ml_count} patterns",
            "",
            "   Manual pattern statistics:",
            f"   - Average Sharpe: {result.comparison.manual_avg_sharpe:.2f}",
            f"   - Average win rate: {result.comparison.manual_avg_win_rate * 100:.1f}%",
            "",
            "   ML pattern statistics:",
            f"   - Average Sharpe: {result.comparison.ml_avg_sharpe:.2f}",
            f"   - Average win rate: {result.comparison.ml_avg_win_rate * 100:.1f}%",
            "",
            "   Improvement metrics:",
            f"   - Sharpe improvement: {result.comparison.sharpe_improvement * 100:+.1f}% {status_symbol}",
            f"   - Win rate improvement: {result.comparison.win_rate_improvement * 100:+.1f}%",
            "",
            f"   Status: {status_symbol if result.comparison.ml_superior else '✗'} "
            f"{'PASS (ML patterns superior)' if result.comparison.ml_superior else 'FAIL (ML patterns inferior)'}",
            "",
        ]
    )

    # Recommendations
    report_lines.extend(
        [
            "RECOMMENDATIONS",
            "─" * 15,
            "",
        ]
    )

    if result.all_success:
        report_lines.extend(
            [
                "1. Deploy with confidence",
                "   - All success criteria exceeded",
                "   - Patterns show excellent generalization",
                "   - Diversity across regimes ensures adaptability",
                "",
                "2. Integration strategy",
                "   - Integrate into ConfluenceEngine immediately",
                "   - Weight ML patterns slightly lower initially (0.85x)",
                "   - Monitor performance daily for first month",
                "",
                "3. Continuous learning",
                "   - Weekly re-clustering active (Task 022)",
                "   - Expect 2-3 new patterns weekly",
                "   - Prune underperformers monthly",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "1. Review failures",
            ]
        )
        if not result.pattern_count.success:
            report_lines.append(
                "   - Insufficient patterns discovered - review ML parameters"
            )
        if not result.profitability.success:
            report_lines.append("   - Low profitability rate - improve pattern quality")
        if not result.diversity.success:
            report_lines.append("   - Insufficient diversity - expand training data")
        if not result.generalization.success:
            report_lines.append("   - Poor generalization - risk of overfitting")
        if not result.comparison.ml_superior:
            report_lines.append(
                "   - ML underperforms manual - review discovery methodology"
            )
        report_lines.append("")
        report_lines.extend(
            [
                "2. Do NOT deploy",
                "   - Critical criteria not met",
                "   - Return to pattern discovery phase",
                "",
            ]
        )

    # Conclusion
    report_lines.extend(
        [
            "CONCLUSION",
            "─" * 10,
        ]
    )

    if result.all_success:
        report_lines.extend(
            [
                "ML Pattern Discovery system VALIDATED successfully.",
                "Ready for production deployment and integration",
                "with RL training system.",
            ]
        )
    else:
        report_lines.extend(
            [
                "ML Pattern Discovery system FAILED validation.",
                "Review failures and return to discovery phase.",
                "Do NOT proceed to production deployment.",
            ]
        )

    # Join all lines
    return "\n".join(report_lines)
