import logging

import a_follow_the_brick as f


def reading_text(reading):
    if not isinstance(reading, dict):
        return "reading=N/A"

    def fmt(key, sign=False):
        try:
            val = float(reading.get(key))
            return f"{val:+.1f}" if sign else f"{val:.1f}"
        except Exception:
            return "N/A"

    return (
        f"dist={fmt('dist_mm')}mm "
        f"x={fmt('x_mm', True)}mm "
        f"y={fmt('y_mm', True)}mm "
        f"conf={fmt('confidence_pct')}%"
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    vision = None
    robot = None
    try:
        print(
            "[CONTINUE] Starting from current pose: empty Step 2 -> Step 3 -> "
            "Step 4 -> reset -> holding game.",
            flush=True,
        )
        f._set_game_profile("empty")
        vision = f.BrickDetector(debug=True)
        vision.set_runtime_tuning(**dict(f.CROWN_PROFILE_TUNING))
        f._warmup(vision)
        robot = f.Robot()

        pre = f._wait_for_confident_brick(vision)
        print("[CONTINUE] Pregame " + reading_text(pre), flush=True)
        if not bool(pre.get("confident")):
            print("[CONTINUE] Pregame visibility failed; parked without motion.", flush=True)
            return 1

        step2 = f._run_step2_settle_sequence(vision, robot)
        print(
            "[CONTINUE] Empty S2 settle: "
            f"success={bool(step2.get('success'))} "
            f"target_met={bool(step2.get('target_met'))} "
            f"reason={step2.get('reason')} "
            f"precision={step2.get('precision_counts')} "
            f"{reading_text(step2.get('reading'))}",
            flush=True,
        )
        if not (bool(step2.get("success")) and bool(step2.get("target_met"))):
            print("[CONTINUE] Empty S2 not honestly won; parked.", flush=True)
            return 1

        step3 = f._run_step3_seat_sequence(vision, robot)
        print(
            "[CONTINUE] Empty S3 seat: "
            f"success={bool(step3.get('success'))} "
            f"target_met={bool(step3.get('target_met'))} "
            f"reason={step3.get('reason')} "
            f"precision={step3.get('precision_counts')} "
            f"{reading_text(step3.get('reading'))}",
            flush=True,
        )
        if not (bool(step3.get("success")) and bool(step3.get("target_met"))):
            print("[CONTINUE] Empty S3 not honestly won; parked.", flush=True)
            return 1

        step4 = f._run_step3_lift_sequence(vision, robot)
        print(
            "[CONTINUE] Empty S4 lift: "
            f"success={bool(step4.get('success'))} "
            f"target_met={bool(step4.get('target_met'))} "
            f"holding={bool(step4.get('holding'))} "
            f"reason={step4.get('reason')} {reading_text(step4.get('reading'))}",
            flush=True,
        )
        if bool(step4.get("holding")):
            f._set_game_profile("holding")
        if not (
            bool(step4.get("success"))
            and (bool(step4.get("target_met")) or bool(step4.get("fallback_reset_ok")))
        ):
            print("[CONTINUE] Empty S4 not honestly won; parked.", flush=True)
            return 1

        reset = f._run_reset_sequence(vision, robot)
        print(
            "[CONTINUE] Reset after empty game: "
            f"success={bool(reset.get('success'))} "
            f"reason={reset.get('reason')} {reading_text(reset.get('reading'))}",
            flush=True,
        )
        if not bool(reset.get("success")):
            print("[CONTINUE] Reset after empty game failed; parked.", flush=True)
            return 1

        f._set_game_profile("holding")
        print("[CONTINUE] Entering holding game loop. Holding S2 drop is now 1400ms.", flush=True)
        stats = f._follow_loop(
            vision,
            robot,
            duration_s=3600.0,
            reset_after_win=True,
            stop_after_win=False,
            stop_after_step2=False,
            step2_probe_before_forward=False,
            debug_mode=False,
            max_cycles=10,
        )
        print("[CONTINUE] Holding/empty loop ended: " + str(stats.get("last_action")), flush=True)
        print("[RESULTS]", flush=True)
        print(f._format_game_results_table(stats), flush=True)
        return 0
    except KeyboardInterrupt:
        print("[CONTINUE] Interrupted.", flush=True)
        return 130
    except Exception as exc:
        print("[CONTINUE] Failed gracefully: " + repr(exc), flush=True)
        return 1
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
            try:
                robot.close()
            except Exception:
                pass
        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass
        print("[CONTINUE] Closed robot/camera resources.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
