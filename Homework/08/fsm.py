from dataclasses import dataclass, field
from typing import Optional


@dataclass
class State:
    is_terminal: bool
    transitions: dict[str, "State"] = field(default_factory=dict)

    def add_transition(self, char, state):
        self.transitions[char] = state


class FSM:
    def __init__(self, states: list[State], initial: int):
        self.states = states
        self.initial = initial

    def is_terminal(self, state_id):
        return self.states[state_id].is_terminal

    def move(self, line: str, start: Optional[int] = None) -> Optional[int]:
        if start is None:
            start = self.initial
        next = self.states[start]
        for state in line:
            next = next.transitions.get(state, None)
            if next is None:
                return None
        return self.states.index(next)


    def accept(self, candidate: str) -> bool:
        """Check if the candidate is accepted by the FSM.

        Args:
            candidate (str): line to check
        Returns:
            is_accept (bool): result of checking
        """
        next = self.move(candidate)
        if next is not None and self.states[next].is_terminal:
            return True
        return False

    def validate_continuation(self, state_id: int, continuation: str) -> bool:
        """Check if the continuation can be achieved from the given state.

        Args:
            state_id (int): state to iterate from
            continuation (str): continuation to check
        Returns:
            is_possible (bool): result of checking
        """
        if self.move(continuation, state_id) is not None:
            return True
        return False


def build_odd_zeros_fsm() -> tuple[FSM, int]:
    """FSM that accepts binary numbers with odd number of zeros

    For example,
    - correct words: 0, 01, 10, 101010
    - incorrect words: 1, 1010

    Args:
    Returns:
        fsm (FSM): FSM
        start_state (int): index of initial state
    """
    false_state = State(False)
    true_state = State(True)
    false_state.add_transition('0', true_state)
    false_state.add_transition('1', false_state)
    true_state.add_transition('0', false_state)
    true_state.add_transition('1', true_state)
    fsm = FSM([false_state, true_state], 0)
    return fsm, 0



if __name__ == "__main__":
    _fsm, _ = build_odd_zeros_fsm()
    print("101010 -- ", _fsm.accept("101010"))
    print("10101 -- ", _fsm.accept("10101"))
