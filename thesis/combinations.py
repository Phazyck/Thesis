"""
A tiny module for producing all
possible combinations of a list of elements.
"""


def combinations(k):
    """
    A function for producing all
    possible combinations of a list of elements.
    """
    def combis(elements):
        """
        A recursive helper-function.
        """
        if elements == []:
            return []
        else:
            head = elements[-1]
            tail = elements[:-1]

            part1 = combis(tail)
            part2 = [l + [head] for l in part1]
            part3 = [[head]]
            return part1 + part2 + part3

    combination_tuples = [tuple(l) for l in combis(k)]

    combination_tuples.sort(key=lambda t: (len(t), t))
    return combination_tuples
