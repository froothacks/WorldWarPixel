"""my-project."""


def greet(name: str = "world") -> str:
    return f"Hello, {name}!"


def main() -> None:
    print(greet())
