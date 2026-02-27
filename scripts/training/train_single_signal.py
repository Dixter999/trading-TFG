#!/usr/bin/env python3
"""
Train Single Signal - Phase 3-5 Executor

Runs Optuna tuning (Phase 3), 30-fold training (Phase 4), and test validation (Phase 5)
for a SPECIFIC signal that passed Phase 2.

Usage:
    python scripts/train_single_signal.py \
        --symbol EURUSD \
        --direction long \
        --signal-name Triple_momentum_long \
        --timeframe H4 \
        --optuna-trials 50

Features:
    - Loads training/test data splits from Phase 1-2
    - Uses signal definition from comprehensive discovery
    - Updates pipeline tracking JSON after each phase
    - Full audit trail maintained
"""

import argparse
import sys
import logging
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_tracking import (
    update_phase3,
    update_phase4,
    update_phase5,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_signal_from_discovery(symbol: str, direction: str, signal_name: str, timeframe: str):
    """
    Load signal definition from comprehensive discovery results.

    Args:
        symbol: Trading symbol
        direction: Signal direction
        signal_name: Name of signal to load
        timeframe: Signal timeframe

    Returns:
        Signal function from indicator_discovery.signals
    """
    from run_hybrid_v4_pipeline import _create_signals_for_timeframe

    # Generate all signals for this direction and timeframe
    all_signals = _create_signals_for_timeframe(direction, timeframe)

    # Find the signal by name
    for sig in all_signals:
        if sig.name == signal_name:
            return sig

    raise ValueError(
        f"Signal {signal_name} not found in discovery library. "
        f"Available signals: {[s.name for s in all_signals[:10]]}... ({len(all_signals)} total)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train single signal through Phase 3-5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train EURUSD LONG Triple_momentum_long with 50 Optuna trials
  python scripts/train_single_signal.py --symbol EURUSD --direction long \\
      --signal-name Triple_momentum_long --timeframe H4 --optuna-trials 50

  # Skip Optuna (use defaults) and train directly
  python scripts/train_single_signal.py --symbol GBPUSD --direction short \\
      --signal-name SMA50_200_RSI_Stoch_short --timeframe H4 --skip-optuna

  # Run only Phase 3 (Optuna tuning)
  python scripts/train_single_signal.py --symbol EURUSD --direction long \\
      --signal-name Triple_momentum_long --timeframe H4 --phase3-only
        """,
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (EURUSD, GBPUSD, etc.)",
    )

    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["long", "short"],
        help="Signal direction",
    )

    parser.add_argument(
        "--signal-name",
        type=str,
        required=True,
        help="Name of signal to train (must have passed Phase 2)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        required=True,
        help="Timeframe for signal (M30, H1, H2, H4, H8, H12, D1)",
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )

    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Skip Optuna tuning, use default hyperparameters",
    )

    parser.add_argument(
        "--optuna-folds",
        type=int,
        default=5,
        help="Number of folds for Optuna cross-validation (default: 5). Use 3 for faster tuning.",
    )

    parser.add_argument(
        "--no-d1",
        action="store_true",
        help="Disable D1/multi-TF features for 6x faster training. Use for high-WR signals.",
    )

    parser.add_argument(
        "--phase3-only",
        action="store_true",
        help="Run only Phase 3 (Optuna), stop before Phase 4",
    )

    parser.add_argument(
        "--phase4-only",
        action="store_true",
        help="Run only Phase 4 (30-fold), skip Phase 3 (requires existing Optuna results)",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=30,
        help="Number of folds for Phase 4 training (default: 30)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/indicators"),
        help="Data directory with CSV files",
    )

    parser.add_argument(
        "--smart-loading",
        action="store_true",
        default=False,
        help="Use smart loading to only load signal-specific columns (reduces memory/I/O)",
    )

    args = parser.parse_args()

    # Normalize inputs (lowercase for tracking, uppercase for display)
    symbol = args.symbol.lower()
    direction = args.direction.lower()
    signal_name = args.signal_name

    # Issue #530: Compute use_d1_features flag
    use_d1_features = not args.no_d1

    print("\n" + "=" * 70)
    print(f"TRAIN SINGLE SIGNAL: {symbol.upper()} {direction.upper()} - {signal_name}")
    print("=" * 70)
    print(f"Timeframe: {args.timeframe}")
    print(f"Optuna Trials: {args.optuna_trials if not args.skip_optuna else 'SKIPPED'}")
    print(f"Optuna Folds: {args.optuna_folds}")
    print(f"Training Folds: {args.n_folds}")
    print(f"D1/Multi-TF: {'ENABLED' if use_d1_features else 'DISABLED (6x faster)'}")
    print(f"Smart Loading: {'ENABLED' if args.smart_loading else 'DISABLED'}")
    print("=" * 70)

    # =========================================================================
    # Load Data and Signal
    # =========================================================================
    print("\n[1/4] Loading data and signal...")

    # Load data splits from Phase 1-2
    from run_hybrid_v4_pipeline import (
        load_market_data,
        load_market_data_smart,
        run_phase0_split,
    )

    # Load signal definition FIRST (needed for smart loading)
    try:
        signal = load_signal_from_discovery(symbol, direction, signal_name, args.timeframe)
        logger.info(f"Loaded signal: {signal.name}")
    except Exception as e:
        print(f"❌ Failed to load signal: {e}")
        return 1

    # Now load data (with smart loading if enabled and signal has required_columns)
    try:
        use_smart_loading = (
            args.smart_loading
            and hasattr(signal, 'required_columns')
            and signal.required_columns
        )

        if use_smart_loading:
            # Smart loading: only load columns needed by the signal
            df = load_market_data_smart(
                symbol=symbol,
                signal=signal,
                data_dir=args.data_dir,
                timeframe=args.timeframe,  # Filter by timeframe (CRITICAL for EURUSD)
            )
            logger.info(f"Smart loading: {len(df.columns)} columns loaded")
            print(f"✅ Smart loaded {len(df)} candles ({len(df.columns)} columns)")
        else:
            # Full loading: load all columns (backward compatible)
            df = load_market_data(
                symbol=symbol,
                data_dir=args.data_dir,
                timeframe=args.timeframe,  # Filter by timeframe (CRITICAL for EURUSD)
            )
            if args.smart_loading:
                logger.info(
                    f"Smart loading requested but signal has no required_columns, "
                    f"falling back to full load ({len(df.columns)} columns)"
                )
            else:
                logger.info(f"Full loading: {len(df.columns)} columns loaded")
            print(f"✅ Loaded {len(df)} candles ({len(df.columns)} columns)")

        split_result = run_phase0_split(df, symbol)
        train_df = split_result["train_df"]
        test_df = split_result["test_df"]
        filtered_df = split_result["filtered_df"]  # 3-year filtered for consistent splits

        print(f"✅ Data split: {len(train_df)} train, {len(test_df)} test")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1

    print(f"✅ Loaded signal: {signal.name}")

    # =========================================================================
    # Phase 3: Optuna Hyperparameter Tuning
    # =========================================================================
    best_params = None

    if not args.skip_optuna and not args.phase4_only:
        print("\n[2/4] Running Phase 3: Optuna Hyperparameter Tuning...")
        update_phase3(symbol, direction, signal_name, started=True)

        from run_hybrid_v4_pipeline import run_phase3_optuna

        try:
            optuna_result = run_phase3_optuna(
                train_df=train_df,
                signal=signal,
                symbol=symbol,
                direction=direction,
                n_trials=args.optuna_trials,
                n_folds=args.optuna_folds,  # Issue #530: Configurable folds
                use_d1_features=use_d1_features,  # Issue #530: Optionally disable for speed
            )

            if optuna_result["passed"]:
                best_params = optuna_result["best_params"]
                print(f"✅ Phase 3 Complete: Best PF = {optuna_result['best_value']:.2f}")
                print(f"   Best Params: {best_params}")

                update_phase3(
                    symbol,
                    direction,
                    signal_name,
                    completed=True,
                    best_params=best_params,
                    best_pf=optuna_result["best_value"],
                    trials_completed=args.optuna_trials,
                )
            else:
                print("⚠️  Phase 3 Failed: Using default hyperparameters")
                best_params = {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "n_steps": 1024,
                    "batch_size": 128,
                }

                update_phase3(
                    symbol,
                    direction,
                    signal_name,
                    completed=True,
                    best_params=best_params,
                    best_pf=0.0,
                    trials_completed=0,
                )

        except Exception as e:
            print(f"❌ Phase 3 Failed: {e}")
            best_params = {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "n_steps": 1024,
                "batch_size": 128,
            }

        if args.phase3_only:
            print("\n✅ Phase 3 Complete (phase3-only mode)")
            print("   Run again without --phase3-only to continue to Phase 4")
            return 0

    elif args.skip_optuna or args.phase4_only:
        print("\n[2/4] Phase 3: SKIPPED (using default hyperparameters)")
        best_params = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "n_steps": 1024,
            "batch_size": 128,
        }

    # =========================================================================
    # Phase 4: 30-Fold Training
    # =========================================================================
    print("\n[3/4] Running Phase 4: 30-Fold Training...")
    update_phase4(symbol, direction, signal_name, started=True)

    from run_hybrid_v4_pipeline import run_phase4_training

    try:
        training_result = run_phase4_training(
            train_df=train_df,
            signal=signal,
            symbol=symbol,
            direction=direction,
            best_params=best_params,
            n_folds=args.n_folds,
            use_d1_features=use_d1_features,  # Issue #530: Pass through
            discovered_signal=signal,  # Issue #530: Use same signal as Phase 3
        )

        if training_result["passed"]:
            print(f"✅ Phase 4 Complete: {training_result['successful_folds']}/{args.n_folds} folds")
            print(f"   Average PF: {training_result['avg_pf']:.2f}")
            print(f"   Average WR: {training_result['avg_wr']:.1f}%")

            update_phase4(
                symbol,
                direction,
                signal_name,
                completed=True,
                folds_completed=training_result["successful_folds"],
                avg_pf=training_result["avg_pf"],
                avg_wr=training_result["avg_wr"],
            )
        else:
            print("❌ Phase 4 Failed: Not enough successful folds")
            update_phase4(
                symbol,
                direction,
                signal_name,
                completed=False,
                folds_completed=training_result.get("successful_folds", 0),
                avg_pf=0.0,
                avg_wr=0.0,
            )
            return 1

    except Exception as e:
        print(f"❌ Phase 4 Failed: {e}")
        update_phase4(
            symbol,
            direction,
            signal_name,
            completed=False,
            folds_completed=0,
            avg_pf=0.0,
            avg_wr=0.0,
        )
        return 1

    # =========================================================================
    # Phase 5: Final Test Validation
    # =========================================================================
    print("\n[4/4] Running Phase 5: Final Test Validation...")
    update_phase5(symbol, direction, signal_name, started=True)

    from run_hybrid_v4_pipeline import run_phase5_validation

    try:
        test_result = run_phase5_validation(
            full_df=filtered_df,
            model_paths=training_result["model_paths"],
            symbol=symbol,
            direction=direction,
            training_avg_pf=training_result["avg_pf"],
            use_d1_features=use_d1_features,  # Issue #530: Must match training
            discovered_signal=signal,  # Issue #530: Must match training
        )

        if test_result["passes_production"]:
            print(f"✅ Phase 5 Complete: PASSED!")
            print(f"   Test PF: {test_result['ensemble_pf']:.2f}")
            print(f"   Test WR: {test_result['ensemble_wr']:.1f}%")

            update_phase5(
                symbol,
                direction,
                signal_name,
                completed=True,
                test_pf=test_result["ensemble_pf"],
                test_wr=test_result["ensemble_wr"],
                passed=True,
            )
        else:
            print(f"⚠️  Phase 5 Complete: Did not meet production criteria")
            print(f"   Test PF: {test_result.get('ensemble_pf', 0):.2f}")
            print(f"   Test WR: {test_result.get('ensemble_wr', 0):.1f}%")
            if test_result.get("rejection_reasons"):
                print(f"   Reasons: {', '.join(test_result['rejection_reasons'])}")

            update_phase5(
                symbol,
                direction,
                signal_name,
                completed=True,
                test_pf=test_result.get("ensemble_pf", 0),
                test_wr=test_result.get("ensemble_wr", 0),
                passed=False,
            )

    except Exception as e:
        print(f"❌ Phase 5 Failed: {e}")
        update_phase5(
            symbol,
            direction,
            signal_name,
            completed=False,
            test_pf=0.0,
            test_wr=0.0,
            passed=False,
        )
        return 1

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE FOR SINGLE SIGNAL")
    print("=" * 70)
    print(f"Signal: {signal_name} ({args.timeframe})")
    print(f"Symbol: {symbol} {direction.upper()}")
    print(f"Phase 3: {'✅ Completed' if best_params else '⚠️ Skipped'}")
    print(f"Phase 4: ✅ {training_result['successful_folds']}/{args.n_folds} folds")
    print(f"Phase 5: {'✅ PASSED' if test_result['passes_production'] else '⚠️ Below threshold'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
