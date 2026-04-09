import numpy as np
from PIL import Image
import os

# --- NÚCLEO MATEMÁTICO (FROM SCRATCH) ---

def manual_conv(image, kernel):
    """Realiza a convolução sem funções prontas."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    # Padding manual para manter dimensões
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)
    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)
    return output

def get_sobel(img):
    """Detector de Sobel: enfatiza bordas com pesos maiores no centro[cite: 46]."""
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = manual_conv(img, sx)
    gy = manual_conv(img, sy)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag / (mag.max() if mag.max() > 0 else 1) * 255).astype(np.uint8)

def get_prewitt(img):
    """Detector de Prewitt: usa pesos uniformes no kernel."""
    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gx = manual_conv(img, px)
    gy = manual_conv(img, py)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag / (mag.max() if mag.max() > 0 else 1) * 255).astype(np.uint8)

def get_canny(img):
    """Implementação completa do algoritmo de Canny[cite: 17, 18, 45]."""
    # 1. Filtro Gaussiano (Redução de ruído) [cite: 16, 17]
    gauss = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], 
                      [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273
    smoothed = manual_conv(img, gauss)
    
    # 2. Gradiente (Sobel interno)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = manual_conv(smoothed, sx)
    gy = manual_conv(smoothed, sy)
    mag = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx) * 180 / np.pi
    theta[theta < 0] += 180
    
    # 3. Supressão Não-Máxima (Bordas finas) [cite: 18, 19]
    nms = np.zeros_like(mag)
    h, w = mag.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = theta[i,j]
            q, r = 255, 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = mag[i, j+1], mag[i, j-1]
            elif (22.5 <= angle < 67.5):
                q, r = mag[i+1, j-1], mag[i-1, j+1]
            elif (67.5 <= angle < 112.5):
                q, r = mag[i+1, j], mag[i-1, j]
            elif (112.5 <= angle < 157.5):
                q, r = mag[i-1, j-1], mag[i+1, j+1]
            if mag[i,j] >= q and mag[i,j] >= r: nms[i,j] = mag[i,j]

    # 4. Histerese (Limiarização)
    high = nms.max() * 0.2
    low = high * 0.5
    res = np.zeros_like(nms, dtype=np.uint8)
    res[nms >= high] = 255
    res[(nms >= low) & (nms < high)] = 50
    return res

# --- GERENCIADOR DE ARQUIVOS ---

def processar_imagens(lista_arquivos, pasta_origem, pasta_destino, eh_rgb=False):
    """Processa e salva cada detector separadamente para comparação[cite: 24, 42]."""
    if not os.path.exists(pasta_destino): os.makedirs(pasta_destino)
    
    for nome_arq in lista_arquivos:
        img_pil = Image.open(os.path.join(pasta_origem, nome_arq))
        base_name = os.path.splitext(nome_arq)[0]
        
        # Lógica de processamento (RGB ou Cinza)
        if eh_rgb:
            canais = [np.array(c, dtype=float) for c in img_pil.split()]
            # Aplica em cada canal e faz o merge (Ponto Extra [cite: 42])
            f_canny = np.maximum.reduce([get_canny(c) for c in canais])
            f_sobel = np.maximum.reduce([get_sobel(c) for c in canais])
            f_prewitt = np.maximum.reduce([get_prewitt(c) for c in canais])
        else:
            img_arr = np.array(img_pil.convert('L'), dtype=float)
            f_canny = get_canny(img_arr)
            f_sobel = get_sobel(img_arr)
            f_prewitt = get_prewitt(img_arr)
            
        # Salvando cada um individualmente para o relatório
        Image.fromarray(f_canny).save(os.path.join(pasta_destino, f"{base_name}_CANNY.jpg"))
        Image.fromarray(f_sobel).save(os.path.join(pasta_destino, f"{base_name}_SOBEL.jpg"))
        Image.fromarray(f_prewitt).save(os.path.join(pasta_destino, f"{base_name}_PREWITT.jpg"))
        
        print(f"Finalizado: {nome_arq} -> Gerados arquivos individuais Canny, Sobel e Prewitt.")

# --- EXECUÇÃO ---

fotos_pb = ['foto_1.jpg']
fotos_coloridas = ['colorida_1.jpg']

processar_imagens(fotos_pb, 'fotos', 'resultados2', eh_rgb=False)
processar_imagens(fotos_coloridas, 'fotosrgb', 'resultadosrgb2', eh_rgb=True)