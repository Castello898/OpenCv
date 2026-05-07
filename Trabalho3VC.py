import cv2
import numpy as np
import os

def detect_circles_in_folder(input_folder="fotos", output_folder="resultados3"):
    # 1. Cria a pasta de resultados se ela não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta '{output_folder}' criada com sucesso.")

    # 2. Verifica se a pasta de fotos existe
    if not os.path.exists(input_folder):
        print(f"Erro: A pasta '{input_folder}' não foi encontrada.")
        print(f"Por favor, crie uma pasta chamada '{input_folder}' no mesmo local deste script e coloque suas imagens lá.")
        return

    # 3. Percorre todos os arquivos dentro da pasta de entrada
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        
        # Carregar a imagem
        img = cv2.imread(image_path)
        
        
        # Pula arquivos que não são imagens válidas
        if img is None:
            print(f"Ignorando arquivo não reconhecido como imagem: {filename}")
            continue

        print(f"\nProcessando: {filename}...")

        # Pré-processamento: Tons de cinza e Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        # Aplicar a Transformada de Hough para Círculos
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=2,
            param1= 20,
            param2=15,
            minRadius=2,
            maxRadius=15
        )

        # Desenhar os resultados
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Contorno do círculo (verde)
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Centro do círculo (vermelho)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                
            print(f"-> {len(circles[0])} círculo(s) detectado(s).")
        else:
            print("-> Nenhum círculo detectado.")

        # 4. Salvar a imagem processada na pasta de saída
        output_path = os.path.join(output_folder, f"resultado_{filename}")
        cv2.imwrite(output_path, img)
        print(f"-> Imagem salva em: {output_path}")

# Executar a função
detect_circles_in_folder()