import cv2
import numpy as np
import glob
import os

# ====================================================================
# CONFIGURAÇÃO DE PASTAS
# ====================================================================
pasta_entrada = "azul/*.jpg" # Pode mudar para png, jpeg se precisar
pasta_saida = "azul_normalizada"

# Se a pasta de saída não existir, o Python cria ela para você automaticamente
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

print("--- INICIANDO PROCESSAMENTO EM LOTE ---")

# O glob.glob cria uma lista com o caminho de todas as fotos da pasta
for caminho_foto in glob.glob(pasta_entrada):
    # Extrai só o nome do arquivo (ex: "IMG_4864.jpg") para log e para salvar
    nome_arquivo = os.path.basename(caminho_foto)
    print(f"\nProcessando: {nome_arquivo}")
    
    # 0. LÊ A IMAGEM
    img_original = cv2.imread(caminho_foto)
    if img_original is None:
        print("  - Erro ao ler imagem.")
        continue
        
    h_orig, w_orig = img_original.shape[:2]

    # 1. CORTE INTELIGENTE (ROI)
    topo, baixo = int(h_orig * 0.21), int(h_orig * 0.80)
    esq, dir_   = int(w_orig * 0.29), int(w_orig * 0.81)

    img = img_original[topo:baixo, esq:dir_]
    h, w = img.shape[:2]

    # 2. RUÍDO DO EVA
    meio_y, meio_x = h // 2, w // 2
    recorte_eva = img[meio_y:meio_y+100, meio_x-150:meio_x-50]
    ruido = np.std(cv2.cvtColor(recorte_eva, cv2.COLOR_BGR2GRAY))
    print(f"  - Ruído do EVA: {ruido:.2f}")

    # ====================================================================
    # FATIAMENTO DE CORES (HSV)
    # ====================================================================
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. CARTÃO BRANCO
    cnts_cartao, _ = cv2.findContours(cv2.inRange(img_hsv, (0, 0, 150), (179, 80, 255)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cartao = next((c for c in sorted(cnts_cartao, key=cv2.contourArea, reverse=True) if cv2.contourArea(c) < (h*w*0.10)), None)

    if cartao is not None:
        xc, yc, wc, hc = cv2.boundingRect(cartao)
        mx, my = int(wc*0.2), int(hc*0.2)
        b, g, r, _ = cv2.mean(img[yc+my : yc+hc-my, xc+mx : xc+wc-mx])
        print(f"  - Cor Cartão: Azul={b:.2f} | Verde={g:.2f} | Vermelho={r:.2f}")
    else:
        print("  - Erro: Cartão não encontrado.")

    # 4. MOEDA ESCURA E NORMALIZAÇÃO
    cnts_moeda, _ = cv2.findContours(cv2.inRange(img_hsv, (0, 30, 0), (40, 255, 150)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    moeda = next((c for c in sorted(cnts_moeda, key=cv2.contourArea, reverse=True) if 500 < cv2.contourArea(c) < 25000), None)

    if moeda is not None:
        _, raio = cv2.minEnclosingCircle(moeda)
        
        # A moeda de 25 centavos tem 25.0 milímetros
        diametro_px = raio * 2
        escala_atual_px_mm = diametro_px / 25.0 
        
        # O Padrão: Forçar todas as imagens a terem exatos 5 pixels por milímetro
        fator_correcao = 5.0 / escala_atual_px_mm
        img_padronizada = cv2.resize(img, None, fx=fator_correcao, fy=fator_correcao)
        
        print(f"  - Moeda: {escala_atual_px_mm:.2f} px/mm -> Fator Zoom: {fator_correcao:.2f}x")
        
        # SALVA A IMAGEM NORMALIZADA NA PASTA NOVA
        caminho_salvar = os.path.join(pasta_saida, f"norm_{nome_arquivo}")
        cv2.imwrite(caminho_salvar, img_padronizada)
        print(f"  - Salvo em: {caminho_salvar}")
    else:
        print("  - Erro: Moeda não encontrada. Imagem não normalizada.")

print("\n--- PROCESSAMENTO CONCLUÍDO ---")