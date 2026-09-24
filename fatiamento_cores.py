import cv2
import numpy as np
import glob
import os

# ====================================================================
# 1. DICIONÁRIO DE CONFIGURAÇÕES (Suas coordenadas calibradas)
# ====================================================================
CONFIG = {
    "azul": {
        "roi_eva":  (0.21, 0.80, 0.29, 0.81),
        "cartao":   (0.29, 0.33, 0.59, 0.68)
    },
    "branco": {
        "roi_eva":  (0.21, 0.80, 0.27, 0.80),
        "cartao":   (0.25, 0.30, 0.63, 0.72)
    },
    "verde": {
        "roi_eva":  (0.24, 0.82, 0.34, 0.85),
        "cartao":   (0.30, 0.34, 0.57, 0.64)
    }
}

# -> DEFINA QUAL COR VOCÊ VAI PROCESSAR AGORA ("azul", "branco" ou "verde")
cor_atual = "verde"

pasta_entrada = f"{cor_atual}/*.jpg"
pasta_saida = f"{cor_atual}_normalizada"

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

print(f"--- INICIANDO PROCESSAMENTO EM LOTE: BASE {cor_atual.upper()} ---")

for caminho_foto in glob.glob(pasta_entrada):
    nome_arquivo = os.path.basename(caminho_foto)
    print(f"\nProcessando: {nome_arquivo}")
    
    img_orig = cv2.imread(caminho_foto)
    if img_orig is None:
        print("  - Erro ao ler imagem.")
        continue
    
    h_orig, w_orig = img_orig.shape[:2]
    cfg = CONFIG[cor_atual]

    # ====================================================================
    # 2. EXTRAÇÃO DIRETA DA COR DO CARTÃO BRANCO
    # ====================================================================
    ct_topo, ct_baixo, ct_esq, ct_dir = cfg["cartao"]
    
    y1_c, y2_c = int(h_orig * ct_topo), int(h_orig * ct_baixo)
    x1_c, x2_c = int(w_orig * ct_esq), int(w_orig * ct_dir)
    
    recorte_cartao = img_orig[y1_c:y2_c, x1_c:x2_c]
    b, g, r, _ = cv2.mean(recorte_cartao)
    print(f"  - Cor do Cartão: Azul={b:.2f} | Verde={g:.2f} | Vermelho={r:.2f}")

    # ====================================================================
    # 3. RECORTE DO EVA E MEDIÇÃO DE RUÍDO
    # ====================================================================
    rt_topo, rt_baixo, rt_esq, rt_dir = cfg["roi_eva"]
    
    img_eva = img_orig[int(h_orig * rt_topo):int(h_orig * rt_baixo), int(w_orig * rt_esq):int(w_orig * rt_dir)]
    h_eva, w_eva = img_eva.shape[:2]

    # Mede o ruído bem no centro do recorte do EVA
    recorte_eva = img_eva[h_eva//2:h_eva//2+100, w_eva//2-150:w_eva//2-50]
    ruido = np.std(cv2.cvtColor(recorte_eva, cv2.COLOR_BGR2GRAY))
    print(f"  - Ruído do EVA: {ruido:.2f}")

    # ====================================================================
    # 4. DETECÇÃO DA MOEDA E NORMALIZAÇÃO ESPACIAL
    # ====================================================================
    img_hsv = cv2.cvtColor(img_eva, cv2.COLOR_BGR2HSV)
    
    # A moeda de bronze (25 centavos) continua sendo marrom e escura
    mascara_moeda = cv2.inRange(img_hsv, np.array([0, 30, 0]), np.array([40, 255, 150]))
    cnts_moeda, _ = cv2.findContours(mascara_moeda, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    moeda = next((c for c in sorted(cnts_moeda, key=cv2.contourArea, reverse=True) if 500 < cv2.contourArea(c) < 25000), None)

    if moeda is not None:
        _, raio = cv2.minEnclosingCircle(moeda)
        
        # Matemática: 25.0 mm de diâmetro físico
        diametro_px = raio * 2
        escala_atual = diametro_px / 25.0 
        
        # Padrão: 5.0 pixels por milímetro
        fator_correcao = 5.0 / escala_atual
        img_padronizada = cv2.resize(img_eva, None, fx=fator_correcao, fy=fator_correcao)
        
        caminho_salvar = os.path.join(pasta_saida, f"norm_{nome_arquivo}")
        cv2.imwrite(caminho_salvar, img_padronizada)
        print(f"  - Moeda: {escala_atual:.2f} px/mm -> Zoom: {fator_correcao:.2f}x. Salvo!")
    else:
        print("  - Erro: Moeda não encontrada. Imagem ignorada.")

print("\n--- PROCESSAMENTO CONCLUÍDO ---")