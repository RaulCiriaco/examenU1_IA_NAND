import random

class Perceptron:
    
    def __init__(self):
        # resultado = x1*w1 + x2*w2 + bias
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.bias = random.uniform(-1, 1)

    def prediccion(self, x1, x2):
        # Calcular la salida de 0 o 1
        resultado = (self.w1 * x1 + self.w2 * x2 + self.bias)

        if resultado >= 0:
            return 1
        else:
            return 0
    
# Funcion de Entrenamiento
def train(perceptron, tablaNand, rangoAprendizaje=0.1, generaciones=20):
    print("\n***ENTRENAMIENTO***\n")
    print("Pesos iniciales: ")
    print("w1: ", round(perceptron.w1,2),
            "w2: ", round(perceptron.w2,2),
            "bias: ", round(perceptron.bias,2)
    )
    # Entrenamiento por generaciones
    for generacion in range(generaciones):
        # recorremos la tabla AND
        for x1, x2, y_real in tablaNand:
            y_prediccion = perceptron.prediccion(x1, x2) # Realiza la operación del Perceptron
            error = y_real - y_prediccion

            perceptron.w1 += rangoAprendizaje * error * x1
            perceptron.w2 += rangoAprendizaje * error * x2
            perceptron.bias += rangoAprendizaje * error

        print(f"Generacion {generacion+1}: "
                f" w1 = {round(perceptron.w1,2)}"
                f" w2 = {round(perceptron.w2,2)}"
                f" bias = {round(perceptron.bias,2)}"
        )

# Funcion de Prueba
def prueba (perceptron, tablaNand):
    print("\n***PRUEBA***\n")
    print("Resultados Finales")

    for x1, x2, _ in tablaNand:
        resultado = perceptron.prediccion(x1, x2)
        print(f"{x1} NAND {x2} = {resultado}")

def main ():
    # MODELO
    tablaNand = [
        # Tabla de verdad para NAND
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0)
    ]

    perceptron = Perceptron()
    train(perceptron, tablaNand, rangoAprendizaje=0.1, generaciones=20)
    prueba(perceptron, tablaNand)
    

if __name__ == "__main__":
    main()
