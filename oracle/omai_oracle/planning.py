"""PDDL instances for the planning cases (IPC 2000 blocks world with four operators, IPC 1998 gripper, and the
Sussman anomaly) and their grounding by pyperplan (github.com/aibasel/pyperplan), the planning oracle."""

import tempfile
from pathlib import Path

from pyperplan.grounding import ground
from pyperplan.pddl.parser import Parser
from pyperplan.task import Operator, Task

BLOCKS = """
(define (domain blocks)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (handempty) (holding ?x))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x)))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y)))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))))
"""
"""The IPC 2000 blocks world (track 1, four operators)."""

GRIPPER = """
(define (domain gripper-strips)
  (:requirements :strips :typing)
  (:types room ball gripper)
  (:predicates (at-robby ?r - room) (at ?b - ball ?r - room) (free ?g - gripper) (carry ?o - ball ?g - gripper))
  (:action move
    :parameters (?from ?to - room)
    :precondition (at-robby ?from)
    :effect (and (at-robby ?to) (not (at-robby ?from))))
  (:action pick
    :parameters (?obj - ball ?room - room ?gripper - gripper)
    :precondition (and (at ?obj ?room) (at-robby ?room) (free ?gripper))
    :effect (and (carry ?obj ?gripper) (not (at ?obj ?room)) (not (free ?gripper))))
  (:action drop
    :parameters (?obj - ball ?room - room ?gripper - gripper)
    :precondition (and (carry ?obj ?gripper) (at-robby ?room))
    :effect (and (at ?obj ?room) (free ?gripper) (not (carry ?obj ?gripper)))))
"""
"""The IPC 1998 gripper domain (typed)."""


def blocks_problem(name: str, blocks: list[str], init: list[str], goal: list[str]) -> str:
    """A blocks-world problem."""
    return (
        f"(define (problem {name}) (:domain blocks) (:objects {' '.join(blocks)})"
        f" (:init {' '.join(init)}) (:goal (and {' '.join(goal)})))"
    )


def gripper_problem(balls: int) -> str:
    """A gripper problem: move every ball from room a to room b."""
    names = [f"ball{i}" for i in range(1, balls + 1)]
    init = ["(at-robby rooma)", "(free left)", "(free right)", *[f"(at {b} rooma)" for b in names]]
    goal = [f"(at {b} roomb)" for b in names]
    return (
        f"(define (problem gripper-{balls}) (:domain gripper-strips)"
        f" (:objects rooma roomb - room {' '.join(names)} - ball left right - gripper)"
        f" (:init {' '.join(init)}) (:goal (and {' '.join(goal)})))"
    )


INSTANCES: list[tuple[str, str, str]] = [
    (
        "sussman",
        BLOCKS,
        blocks_problem(
            "sussman",
            ["a", "b", "c"],
            ["(on c a)", "(ontable a)", "(ontable b)", "(clear c)", "(clear b)", "(handempty)"],
            ["(on a b)", "(on b c)"],
        ),
    ),
    (
        "blocks-4-0",
        BLOCKS,
        blocks_problem(
            "blocks-4-0",
            ["d", "b", "a", "c"],
            [
                "(clear c)",
                "(clear a)",
                "(clear b)",
                "(clear d)",
                "(ontable c)",
                "(ontable a)",
                "(ontable b)",
                "(ontable d)",
                "(handempty)",
            ],
            ["(on d c)", "(on c b)", "(on b a)"],
        ),
    ),
    (
        "blocks-4-1",
        BLOCKS,
        blocks_problem(
            "blocks-4-1",
            ["a", "c", "d", "b"],
            ["(clear b)", "(ontable d)", "(on b c)", "(on c a)", "(on a d)", "(handempty)"],
            ["(on d c)", "(on c a)", "(on a b)"],
        ),
    ),
    (
        "blocks-5-0",
        BLOCKS,
        blocks_problem(
            "blocks-5-0",
            ["b", "e", "a", "c", "d"],
            [
                "(clear a)",
                "(clear c)",
                "(clear d)",
                "(ontable a)",
                "(ontable b)",
                "(ontable d)",
                "(on c e)",
                "(on e b)",
                "(handempty)",
            ],
            ["(on a e)", "(on e b)", "(on b d)", "(on d c)"],
        ),
    ),
    ("gripper-2", GRIPPER, gripper_problem(2)),
    ("gripper-3", GRIPPER, gripper_problem(3)),
]
"""The planning instances: a name, the domain and the problem."""


def operator_name(op: Operator) -> str:
    """An operator's name, the sort key of the operators."""
    return str(op.name)


def task(domain: str, problem: str) -> Task:
    """Parse and ground a PDDL domain and problem with pyperplan, with the operators sorted by name (pyperplan's
    grounding iterates sets, so its operator order, and with it which of several optimal plans a search returns,
    otherwise varies between runs)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "domain.pddl"
        p = Path(tmp) / "problem.pddl"
        d.write_text(domain)
        p.write_text(problem)
        parser = Parser(str(d), str(p))
        dom = parser.parse_domain()
        t = ground(parser.parse_problem(dom))
    t.operators = sorted((op for op in t.operators if op is not None), key=operator_name)
    return t
