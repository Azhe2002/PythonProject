import math


def circle_area(radius: float) -> float:
    """Return the area of a circle with the given radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2


if __name__ == "__main__":
    r = float(input("Enter radius: "))
    print(f"Area: {circle_area(r):.4f}")
