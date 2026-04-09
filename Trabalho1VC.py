import numpy as np
from PIL import Image
import os

# --- OPERAÇÕES MATEMÁTICAS FROM SCRATCH ---

def manual_convolution(image, kernel):
    """Aplica convolução 2D sem utilizar funções externas[cite: 17, 32]."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Padding com zeros para manter o tamanho original
    padded = np.zeros((img_h + 2*pad_h, img_w + 2*pad_w))
    padded[pad_h:-pad_h, pad_w:-pad_w] = image
    
    output = np.zeros_like(image)
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    return output

def get_gaussian_kernel(size=5, sigma=1.4):
    """Gera o kernel de suavização para reduzir ruídos[cite: 16, 17]."""
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return kernel / np.sum(kernel)

def run_canny_logic(img_array):
    """Implementa as etapas de Canny: Gauss, Sobel, NMS e Histerese[cite: 18, 19]."""
    # 1. Suavização
    smooth = manual_convolution(img_array, get_gaussian_kernel())
    
    # 2. Gradientes (Sobel)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = manual_convolution(smooth, sx)
    gy = manual_convolution(smooth, sy)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = (magnitude / magnitude.max() * 255) if magnitude.max() != 0 else magnitude
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle[angle < 0] += 180

    # 3. Supressão Não-Máxima (NMS) - Garante bordas finas [cite: 5, 18]
    h, w = magnitude.shape
    nms = np.zeros((h, w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            ang = angle[i, j]
            q, r = 255, 255
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif (22.5 <= ang < 67.5):
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif (67.5 <= ang < 112.5):
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            elif (112.5 <= ang < 157.5):
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                nms[i, j] = magnitude[i, j]

    # 4. Histerese (Limiarização Dupla)
    high = nms.max() * 0.15
    low = high * 0.3
    res = np.zeros_like(nms, dtype=np.uint8)
    res[nms >= high] = 255
    res[(nms >= low) & (nms < high)] = 50
    
    # Conectividade
    for i in range(1, h-1):
        for j in range(1, w-1):
            if res[i, j] == 50:
                if (res[i-1:i+2, j-1:j+2] == 255).any():
                    res[i, j] = 255
                else:
                    res[i, j] = 0
    return res

# --- PROCESSAMENTO EM LOTE ---

def processar_imagens(lista_arquivos, pasta_origem, pasta_destino, eh_rgb=False):
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        
    for nome_arq in lista_arquivos:
        caminho = os.path.join(pasta_origem, nome_arq)
        img = Image.open(caminho)
        
        if eh_rgb:
            # Processa cada canal separadamente (Ponto Extra) 
            r, g, b = img.split()
            res_r = run_canny_logic(np.array(r, dtype=np.float64))
            res_g = run_canny_logic(np.array(g, dtype=np.float64))
            res_b = run_canny_logic(np.array(b, dtype=np.float64))
            # Combina os canais (OR lógico/Soma)
            final_array = np.maximum(res_r, np.maximum(res_g, res_b))
        else:
            img_gray = img.convert('L')
            final_array = run_canny_logic(np.array(img_gray, dtype=np.float64))
            
        # Salva o resultado
        Image.fromarray(final_array).save(os.path.join(pasta_destino, f"edge_{nome_arq}"))
        print(f"Processado: {nome_arq}")

# --- EXECUÇÃO CONFORME SOLICITADO ---

fotos_pb = ['foto_1.jpg', 'foto_2.jpg', 'foto_3.jpg', 'foto_4.jpg']
fotos_coloridas = ['colorida_1.jpg', 'colorida_2.jpg', 'colorida_3.jpg', 'colorida_4.jpg']

# Processamento P&B
processar_imagens(fotos_pb, 'fotos', 'resultados2', eh_rgb=False)

# Processamento Colorido (Extra)
processar_imagens(fotos_coloridas, 'fotosrgb', 'resultadosrgb2', eh_rgb=True)