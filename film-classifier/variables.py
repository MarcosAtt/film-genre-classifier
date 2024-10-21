genre_name = [
    "crime",
    "romance",
    "science fiction",
    "western",
]

default_Q = 1000
default_seed = 42


def print_avance(i, j, iMax, jMax):
    total = (iMax) * (jMax)
    porcentaje = (i * jMax) + (j)
    if i * j < total:
        print(int(porcentaje / total * 100), "%", end="\r", flush=True)
    if (i + 1) * (j + 1) == total:
        print("100 %!")
