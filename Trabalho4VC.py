import cv2
import numpy as np

def processar_video(caminho_video, num_frames_fundo=50, threshold=30, usar_rgb=False):
    """
    Processa um vídeo para separação de plano de fundo usando três modelos:
    Fixo, Média e Mediana.
    """
    cap = cv2.VideoCapture(caminho_video)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo. Verifique o caminho.")
        return

    frames_fundo = []
    print(f"Coletando os primeiros {num_frames_fundo} frames para modelagem...")
    
    # 1. Coleta de frames para criar os modelos
    for _ in range(num_frames_fundo):
        ret, frame = cap.read()
        if not ret:
            break
            
        if not usar_rgb:
            # Converte para tons de cinza para o requisito padrão
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        frames_fundo.append(frame)
        
    if len(frames_fundo) == 0:
        print("Nenhum frame lido.")
        return

    # Empilha os frames para cálculos estatísticos matriciais
    stack_frames = np.stack(frames_fundo, axis=0)

    # 2. Criação dos Modelos de Fundo
    print("Calculando os modelos de fundo...")
    modelo_fixo = frames_fundo[0] # Usa o primeiro frame como fundo fixo
    modelo_media = np.mean(stack_frames, axis=0).astype(np.uint8)
    modelo_mediana = np.median(stack_frames, axis=0).astype(np.uint8)

    print("Iniciando a subtração (Pressione 'q' para sair)...")
    
    # 3. Subtração de fundo no restante do vídeo
    while True:
        ret, frame_atual = cap.read()
        if not ret:
            print("Fim do vídeo.")
            break
            
        # Guarda o frame original para exibição (Cópia para o relatório)
        frame_display = frame_atual.copy()

        if not usar_rgb:
            frame_processar = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2GRAY)
        else:
            frame_processar = frame_atual

        # Função auxiliar para calcular a diferença absoluta e aplicar o limiar (threshold)
        def subtrair_e_limiar(frame, modelo, thresh_val):
            # Diferença absoluta: D(x,y) = |I(x,y) - B(x,y)|
            diff = cv2.absdiff(frame, modelo)
            
            if usar_rgb:
                # Para RGB, converte a diferença para tons de cinza antes do limiar 
                # ou avalia a magnitude. A conversão é mais simples para criar a máscara binária.
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
            _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            return mask

        # Aplica a subtração para cada modelo
        mask_fixo = subtrair_e_limiar(frame_processar, modelo_fixo, threshold)
        mask_media = subtrair_e_limiar(frame_processar, modelo_media, threshold)
        mask_mediana = subtrair_e_limiar(frame_processar, modelo_mediana, threshold)

        # Redimensiona para caber na tela durante os testes
        escala = 0.5
        def redimensionar(img):
            return cv2.resize(img, (0,0), fx=escala, fy=escala)

        # Exibição dos resultados em tempo real
        cv2.imshow('Original', redimensionar(frame_display))
        cv2.imshow('Mascara - Fundo Fixo', redimensionar(mask_fixo))
        cv2.imshow('Mascara - Modelo Media', redimensionar(mask_media))
        cv2.imshow('Mascara - Modelo Mediana', redimensionar(mask_mediana))

        # Pressione 's' para salvar os frames atuais (útil para o relatório do artigo)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Salvando frames para o relatório...")
            cv2.imwrite("resultado_original.png", frame_display)
            cv2.imwrite("resultado_fixo.png", mask_fixo)
            cv2.imwrite("resultado_media.png", mask_media)
            cv2.imwrite("resultado_mediana.png", mask_mediana)
            cv2.imwrite("fundo_media.png", modelo_media)
            cv2.imwrite("fundo_mediana.png", modelo_mediana)

    cap.release()
    cv2.destroyAllWindows()

# --- Execução Principal ---
if __name__ == "__main__":
    VIDEO_PATH = 'bloons.mp4' # Substitua pelo caminho do seu vídeo
    
    # 1. Execução padrão (Tons de Cinza) - Requisito mínimo
    print("=== TESTE 1: TONS DE CINZA ===")
    processar_video(VIDEO_PATH, num_frames_fundo=60, threshold=30, usar_rgb=False)
    
    # 2. Execução extra (RGB) - Para os 2 pontos extras
    print("\n=== TESTE 2: RGB (PONTOS EXTRAS) ===")
    processar_video(VIDEO_PATH, num_frames_fundo=60, threshold=45, usar_rgb=True)