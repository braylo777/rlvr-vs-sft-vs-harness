import math

import pytest

from rvsh import calc, compute, gsm8k, harness, prompts, rewards


def test_gold_answer_parses_hash_line():
    assert gsm8k.gold_answer("She has 3+4=<<3+4=7>>7 apples.\n#### 7") == "7"
    assert gsm8k.gold_answer("... #### 1,234") == "1234"


def test_normalize_number_variants():
    assert gsm8k.normalize_number("$12.50") == "12.5"
    assert gsm8k.normalize_number("7.0") == "7"
    assert gsm8k.normalize_number("-0") == "0"
    assert gsm8k.normalize_number("abc") is None


def test_extract_answer_priority():
    assert gsm8k.extract_answer("3 + 4 = 7. The answer is 8.") == "8"
    assert gsm8k.extract_answer("so \\boxed{42} done 3") == "42"
    assert gsm8k.extract_answer("total 10 then 12") == "12"
    assert gsm8k.extract_answer("no numbers here") is None


def test_literal_placeholder_falls_back_to_last_number():
    # the base model sometimes prints the template literally
    assert gsm8k.extract_answer("It takes 6 days. The answer is N.") == "6"


def test_is_correct_numeric_equality():
    assert gsm8k.is_correct("The answer is 18.0", "18")
    assert not gsm8k.is_correct("The answer is 17", "18")


def test_solution_to_target_strips_calculator_annotations():
    t = gsm8k.solution_to_target("Half is 48/2=<<48/2=24>>24.\n#### 24")
    assert "<<" not in t and t.endswith("The answer is 24.")


def test_calc_runs_assignment_program():
    assert calc.run_program("a = 48 / 2\nanswer = a + 48") == 72
    assert calc.run_program("```python\nx = 3\ny = x * 4\n```") == 12


def test_calc_rejects_unsafe_code():
    for src in ["import os", "answer = __import__('os')", "answer = open('x')", "answer = (1).real"]:
        with pytest.raises(calc.CalcError):
            calc.run_program(src)


def test_calc_rejects_huge_exponent():
    with pytest.raises(calc.CalcError):
        calc.run_program("answer = 10 ** 1000")


def test_calc_skips_prose_lines():
    assert calc.run_program("Here is code:\nanswer = 2 + 2") == 4


def test_format_number():
    assert calc.format_number(7.0) == "7"
    assert calc.format_number(2.5) == "2.5"


def test_vote_rules():
    assert harness.vote("5", "5", []) == ("5", "agree")
    assert harness.vote("5", "6", ["6", "7"]) == ("6", "majority")
    assert harness.vote("5", "6", [None, None]) == ("6", "tie->pot")
    assert harness.vote(None, None, [None]) == (None, "none")


def test_pot_answer_handles_bad_code():
    assert harness.pot_answer("answer = 1/0") is None
    assert harness.pot_answer("answer = 3 * 4") == "12"


def test_rewards_are_verifiable():
    comps = [[{"role": "assistant", "content": "The answer is 4."}], "I think 5"]
    assert rewards.correctness_reward(comps, ["4", "4"]) == [1.0, 0.0]
    assert rewards.format_reward(comps) == [0.1, 0.0]


def test_compute_ledger_flops():
    led = compute.ComputeLedger(n_params=1_000)
    led.add_inference(10)
    led.add_lora_train(5)
    assert led.flops == 2 * 1000 * 10 + 4 * 1000 * 5


def test_sft_budget_matches_flops():
    n = 357_898_112
    flops = 4.0 * n * 12_345
    assert compute.sft_tokens_for_budget(flops, n) == 12_345


def test_break_even():
    assert compute.break_even_queries(100.0, 10.0) == 10.0
    assert math.isinf(compute.break_even_queries(100.0, 0.0))


def test_usd_cost_uses_assumption():
    one_hour = compute.ASSUMED_GPU["effective_tflops"] * 1e12 * 3600
    assert compute.usd_cost(one_hour) == pytest.approx(compute.ASSUMED_GPU["usd_per_hour"])


def test_prompts_share_system_prompt():
    assert prompts.cot_messages("q")[0]["content"] == prompts.COT_SYSTEM
    assert prompts.fewshot_cot_messages("q")[-1] == {"role": "user", "content": "q"}
    assert len(prompts.pot_messages("q")) == 2 + 2 * len(prompts.POT_SHOTS)
