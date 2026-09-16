import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import maximum_filter
from scipy.integrate import quad
import pandas as pd
from scipy.integrate import quad

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y NUMÉRICOS DEL SISTEMA
# ==============================================================================
N_PUNTOS = 1024          # Resolución de la malla (512x512)
DX = 1e-9               # Resolución espacial: 1 nm por píxel
SIGMA_S = 1.5e-9        # Rugosidad RMS de la superficie metálica: 1.5 nm
XI = 10e-9              # Longitud de correlación espacial: 10 nm
# ==============================================================================
# 2. FUNCIONES MODULARES
# ==============================================================================
def generar_superficie(N,dx,sigma_s,XI):
    #1 Generamos la maya para trabajar con meshgrid
    limite=3 * XI #Generamos el limite espacial del filtro, que es tres veces la longitud de correlación
    coords= np.arange(-limite,limite+dx,dx) #Centramos el vector en el cero 
    X, Y = np.meshgrid(coords, coords)
    #2 Generamos la matriz de ruido de fondo
    ruido= np.random.normal(loc=0.0, scale=1.0, size=(N,N))
    #3 Hacemos el Kernel de correlación normalizado para que la superfice pueda ser suave, y se tenga en cuenta los puntos proximos para calcular las alturas
    kernel= np.exp(-(X**2+Y**2)/XI**2)
    kernel_n=kernel/np.sum(kernel)#La normalización nos asegura que la suma de los pesos sea 1 y evitar alterar la señal original 
    #4 Hacemos la convolución para hacer que los puntos proximos de la matriz se afecten entre ellos
    superficie= fftconvolve(ruido,kernel_n,mode="same")
    #5 Normalizamos para tener una matriz con significado
    superficie_n=((superficie-np.mean(superficie))/np.std(superficie))*sigma_s #Guardamos la infomación que hemos obtenido
    return superficie_n

#Hacemos la función que nos devuelve la posición y el número de vertices de cada poliedro, en esta función no se "forman" todavía las caras
def generar_poblacion_poliedros(rho_granos, L_total, r_medio=18e-9, sigma_r=0.35,media_ba=0.65, sigma_ba=0.12, min_ba=0.35, max_ba=0.90,
media_cb=0.50, sigma_cb=0.10, min_cb=0.30, max_cb=0.70,min_caras=8,max_caras=14):

    """
    Genera una población de M granos de regolito modelados como poliedros convexos irregulares.
    
    Parámetros geométricos y físicos:
    --------------------------------
    rho_granos : float
        Fracción de cobertura objetivo (área proyectada ocupada / área total).
    L_total : float
        Longitud lateral física del dominio en metros.
    r_medio : float
        Radio medio equivalente de la partícula esférica (m).
    sigma_r : float
        Dispersión log-normal del tamaño de grano (Lunar Sourcebook).
    media_ba : float
        Media de la relación de aspecto plano b/a (elongación en el plano xy).
    sigma_ba : float
        Desviación estándar de la relación b/a.
    min_ba, max_ba : float
        Límites de corte para b/a; evita agujas infinitas (<0.35) y círculos perfectos (>0.90).
    media_cb : float
        Media del achatamiento vertical c/b (espesor frente al semieje intermedio).
    sigma_cb : float
        Desviación estándar de la relación c/b.
    min_cb, max_cb : float
        Límites de corte para c/b; garantiza fragmentos laminares y achatados.
    min_caras, max_caras : int
        Rango de vértices aleatorios para el ConvexHull (controla el facetado y la angularidad).
    """
    M = int(rho_granos * L_total**2 / (np.pi * (r_medio**2))) #Calculamos el número de ganos
    #Distibuimos las posiciones de los granos y sus radios correspondientes
    x0 = np.random.uniform(0, L_total, size=M)
    y0 = np.random.uniform(0, L_total, size=M)
    radios = np.random.lognormal(mean=np.log(r_medio), sigma=sigma_r, size=M)

    # Dimensiones con proporciones de esquirla (achatadas en z pero con facetas)
    muestreo_ba = np.random.normal(loc=media_ba, scale=sigma_ba, size=M)#Relación entre a a y b, es decir los ejes del plano de la superficie
    rel_ba = np.clip(muestreo_ba, min_ba, max_ba)
    a = radios / np.sqrt(rel_ba)#Definimos los ejes del elipsoide que será la base para el poliedro 
    b = radios * np.sqrt(rel_ba)
    muestreo_cb = np.random.normal(loc=media_cb, scale=sigma_cb, size=M) #Repetimos lo mismo para el eje vertical 
    rel_cb = np.clip(muestreo_cb, min_cb, max_cb)
    c = b * rel_cb
    angulos = np.random.uniform(0, np.pi, size=M)
    granos = []
    # Generamos los vertices de cada poliedro irregular individual
    for i in range(M):
        n_vertices = np.random.randint(min_caras, max_caras) # Numero de caras/vertices del grano
        # Usamos la distribución normal para distribuir aleatoriamente el número de vertices calculado en la superficie de la forma elipsoidal que hemos creado, así formamos el poliedro cada uno distinto al anterior.
        pts = np.random.normal(size=(n_vertices, 3))
        pts /= np.linalg.norm(pts, axis=1)[:, np.newaxis]
        
        # Escalado: Ajustamos los tamaños del grano para que sean correctos
        pts[:, 0] *= a[i]
        pts[:, 1] *= b[i]
        pts[:, 2] *= c[i]

        # Como el elipsoide se creea en una posición ideal alineada, hay que rotarlo para que cada uno sea distinto y con las puntas apuntando en direciones aleatorias
        th = angulos[i]
        c_th, s_th = np.cos(th), np.sin(th)
        x_rot = pts[:, 0] * c_th - pts[:, 1] * s_th
        y_rot = pts[:, 0] * s_th + pts[:, 1] * c_th
        pts[:, 0], pts[:, 1] = x_rot, y_rot

        granos.append({'pos': (x0[i], y0[i]), 'pts_locales': pts, 'a': a[i], 'b': b[i], 'c': c[i]})

    return granos

def asentar_y_extraer_mallas(granos, Z_metal, N, dx):
    """Calcula el apoyo fisico exacto de cada grano y devuelve matrices 2.5D + caras 3D."""
    Z_top = np.full((N, N), np.nan) #Generamos dos matices de infomación ya que así podemos hacer figuras cerradas sin problemas, las inicializamos como vacias para luego solo llenar los puntos que nos interesan. 
    Z_bot = np.full((N, N), np.nan)
    eje = np.arange(N) * dx #Definimos el eje con el nuemro de puntos y el tamaño
    poliedros_3d = []

    for g in granos:
        #Guardamos la posición de cada grano para usarla al igual que los puntos que tiene en la superficie elipsoidal
        x0, y0 = g['pos'] 
        pts = g['pts_locales'].copy()
        radio_max = np.max(np.hypot(pts[:, 0], pts[:, 1]))

        #Definimos una caja delimitadora para evitar tener cargada infomación más allá de esta. Se hace usando el radio maximo para aseguarnos de dejar dentro todo el poliedro
        j_min, j_max = int(np.clip((x0 - radio_max)//dx, 0, N-1)), int(np.clip((x0 + radio_max)//dx + 2, 0, N))
        i_min, i_max = int(np.clip((y0 - radio_max)//dx, 0, N-1)), int(np.clip((y0 + radio_max)//dx + 2, 0, N))
        if j_min >= j_max or i_min >= i_max: continue

        # Colocamos el regolito sobre el sustrato para que no clipe dentro, es decir, que no quede ni flotando ni incrustado (este proceso no es el que nos dirá el area real de contacto, eso lo haremos acercando las matrices obtenidas con físicas)
        sustrato_local = Z_metal[i_min:i_max, j_min:j_max]
        z_contacto = np.max(sustrato_local)
        z_base_grano = np.min(pts[:, 2])
        
        # Desplazamiento z para contacto perfecto
        pts[:, 0] += x0
        pts[:, 1] += y0
        pts[:, 2] += (z_contacto - z_base_grano)

        # Envolvente poliedrica (Convex Hull) para obtener caras planas reales. Juntamos los vertices para asegurarnos tener poliedros
        hull = ConvexHull(pts)
        caras = [pts[simplex] for simplex in hull.simplices]
        poliedros_3d.append(caras)

        # Mapeo a las matrices 2.5D Z_top y Z_bot para analisis cuantitativo, guardamos la infomación para cuando sea necesaria en un futuro
        X_loc, Y_loc = np.meshgrid(eje[j_min:j_max], eje[i_min:i_max])
        c_hull = ConvexHull(pts[:, :2])
        # Proyeccion en matriz local
        coords_pts = np.vstack((X_loc.ravel(), Y_loc.ravel())).T
        dentro = np.all(np.add(np.dot(coords_pts, c_hull.equations[:, :-1].T), c_hull.equations[:, -1]) <= 1e-12, axis=1)
        dentro = dentro.reshape(X_loc.shape)

        z_t = np.max(pts[:, 2])
        z_b = np.min(pts[:, 2])
        
        # Actualizamos mapas de cota
        Z_top[i_min:i_max, j_min:j_max][dentro] = np.fmax(Z_top[i_min:i_max, j_min:j_max][dentro], z_t)
        Z_bot[i_min:i_max, j_min:j_max][dentro] = np.fmin(Z_bot[i_min:i_max, j_min:j_max][dentro], z_b)

    return Z_top, Z_bot, poliedros_3d



#Para poder verificar AGW es necesario permitir que las dos placas de metal tengan un contacto elástico, con defomraciones. 
#Por simplificación analítica, vamos a ir desplazando el material distancias arbitrarias, que corresponderán a una fuerza, y a un área, luego podremos mirar en las tablas obtenidas para x fuerza, cuanta distancia se desplaza y que area hay. 
#Cabe destacar que en esta parte, en vez de trabajar coin Z1 y Z2 lo hacemos con Z_eq, ya que es lo mismo mirar asperezas con ambas matrices, que con la suma equivalente y una pared plana, lo que computacionalmente es mucho más sencillo  (g(x, y) = z_sup(x, y) - z_inf}(x, y) = (d - Z_2(x, y)) - Z_1(x, y))= d- Z_eq
def caracterizar_asperezas(Z,dx):
    """
    Usamos análisis de los vecinos proximos para saber la posición de los maximos locales, que serán las asperezas a considerar para la mecánica de contacto. Con eso calculamos su radio de curbatura local 
    """
    #Filtramos los maximos locales 
    filtro=maximum_filter(Z,size=3)#Filtramos en un 3x3 de vecinos proximos (Modelo similar a ISING)
    es_pico=(Z==filtro)
    #Excluimos los bordes que no son picos reales
    es_pico[0, :] = es_pico[-1, :] = es_pico[:, 0] = es_pico[:, -1] = False

    alturas_picos=Z[es_pico] #Guradamos la alutra de los maximos
    num_picos=len(alturas_picos)#Guardamos el numero de picos
    area_total = (Z.shape[0] * dx) * (Z.shape[1] * dx)
    rho = num_picos / area_total  # Densidad de picos por m^2

    #Analizamos la curbatura por pico
    i_picos, j_picos = np.where(es_pico)
    d2z_dx2 = (Z[i_picos, j_picos + 1] - 2 * Z[i_picos, j_picos] + Z[i_picos, j_picos - 1]) / (dx**2)
    d2z_dy2 = (Z[i_picos + 1, j_picos] - 2 * Z[i_picos, j_picos] + Z[i_picos - 1, j_picos]) / (dx**2)
    # La curvatura media kappa es -0.5 * laplaciano; el radio es 1 / kappa
    kappa = -0.5 * (d2z_dx2 + d2z_dy2)
    validos = (kappa > 1e-6)#aseguramos que hay curvatura descartando cuando no la haya
    radios = 1/kappa[validos] 
    alturas_picos=alturas_picos[validos]
    R_medio= np.mean(radios)
    sigma_p = np.std(alturas_picos)
    
    return alturas_picos,radios, R_medio, rho ,sigma_p

def barrido_contacto_GW(alturas_picos, radios, E_star, n_pasos=50):
    """
    Calcula la curva de area A_real vs F_N bajando una placa rígida paso a paso. 
    """ 
    #Miramos los puntos maximo y minimos 
    z_max = np.max(alturas_picos)
    z_min = np.min(alturas_picos)
    # Barrido de separación 'd' desde el pico más alto hacia abajo
    separaciones = np.linspace(z_max, z_min + 0.3 * (z_max - z_min), n_pasos) #El 0.3 es la cota de bajada maxima que pude hacer el material, se debe aproximadamente a hacer una resticción de d \approx 2sigma_s o 1.5\sigma_s. (Son valores empiricos de maximo desplazamiento)
    
    #Inicializamos las listas de valores 
    A_real = []
    F_normal = []

    for d in separaciones: 
        #Miramos para un movimiento concreto que particulas están en contacto 
        delta =alturas_picos -d
        en_contacto = delta >0 
        #Para cada altura miramos si hay contacto, si lo hay guardamos la infomación de la fuerza y de las areas de contacto.  
        if np.any(en_contacto):
            delta_c = delta[en_contacto]
            R_c = radios[en_contacto]
            
            # Formulación elástica de Hertz (GW) por cada aspereza
            areas = np.pi * R_c * delta_c #Usamos la fomulación de Hertz para hacer los cálculos
            fuerzas = (4.0 / 3.0) * E_star * np.sqrt(R_c) * (delta_c**1.5) #De igual forma usamos la solución a la fuerza de Hertz para relacionar la presión con la fuerza a través de una integración.
            
            A_real.append(np.sum(areas))
            F_normal.append(np.sum(fuerzas))
        else:
            A_real.append(0.0)
            F_normal.append(0.0)


    return np.array(F_normal), np.array(A_real) , separaciones


#Elaboramos una función que calcule las areas reales analíticas a través de la formulación de GW, así podremos comparar los datos computacionales con los analíticos. 

def calcular_areas_analiticas_GW(df, alturas_picos, R_medio, rho, sigma_p, A_0,E_star):
    """
    Añade al DataFrame la validación teórica de Greenwood-Williamson (GW):
      1. A_GW_integral [nm2]: Solución analítica exacta de GW calculada
         mediante cuadratura numérica (scipy.integrate.quad) sobre una distribución
         gaussiana ideal de alturas de cumbres F_1(h).
      2. A_GW_lineal [nm2]: Límite asintótico analítico de GW para cargas elásticas
         ligeras (A_real = c * (F_N / E*) * sqrt(R / sigma_p), con c = sqrt(pi)).

    Interpretación de discrepancias y validación:
      - Filas iniciales (d >> mean(z)): Errores relativos elevados debidos al
        efecto de tamaño de muestra finito (la superficie discreta NxN trunca la
        cola gaussiana infinita antes de alcanzar cotas extremas).
      - Régimen medio elástico (h in [1.5, 3.0]): Excelente convergencia con la
        aproximación lineal (< 5-8% de error), validando la proporcionalidad A_real ~ F_N.
      - Error residual frente a la integral (~15-20%): Discrepancia física estructural
        esperada. La teoría de GW asume un radio medio esférico idéntico (R_medio) y 
        cumbres sin correlación espacial, mientras que la simulación discreta deforma
        cada aspereza con su curvatura local exacta heterogénea (R_i vía Laplaciano). 
        En general la simulación computacional resulta ser más fidedigna a la realidad
        ya que no considera todas las irregularidades como esferas iguales sino que varía
        su radio de curvatura. 
    """
    #Inicializamos los valores útiles
    z_p_mean = np.mean(alturas_picos)
    phi = lambda s: (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s**2)
    
    A_integral_m2 = []
    F_integral_N = []
    #Vía integración directa: (usamos un for ya que lo hacemos uno a uno) 
    separaciones = df['d [nm]'].values * 1e-9  # m
    for d in separaciones: 
        h = (d - z_p_mean) / sigma_p
        if h < 4.0: #Quitamos los valores muy extremos, más allá de 4 sigma porque computacinalmetne son pesados de calcular
            I_area, _ = quad(lambda s: (s - h) * phi(s), h, 6.0) #Calculamos la integral con quad, usamos un límite de 6 ya que la gausiana a decaido suficiente al ser negativa, y no es necesario ir más allá, pero es el valor sustituitivo de infinito
            a_int = np.pi * rho * A_0 * R_medio * sigma_p * I_area #Se multplica por las constantes para tener el área total
            I_fuerza, _ = quad(lambda s: ((s - h)**1.5) * phi(s), h, 6.0) #Calculamos la integral de la fuerza teórica continua 
            f_int = (4.0 / 3.0) * rho * A_0 * E_star * np.sqrt(R_medio) * (sigma_p**1.5) * I_fuerza
            F_integral_N.append(f_int)
        else:
            a_int = 0.0 # Cuando la cola está muy lejos se indica que es cero
            F_integral_N.append(0.0)
        A_integral_m2.append(a_int)


    #Aproximación lineal usando E* 
    F_integral_N = np.array(F_integral_N)
    # Constante de contacto elastico de GW: sqrt(pi)
    cte_gw = np.sqrt(np.pi)
    A_lineal_m2 = cte_gw * (F_integral_N/ E_star) * np.sqrt(R_medio / sigma_p) # Formula aproximada de hacer la integración gausiana
    
    # Convertimos ambas a nm² y las añadimos al DataFrame
    df['A_GW_integral [nm2]'] = np.array(A_integral_m2) * 1e18
    df['A_GW_lineal [nm2]'] = np.array(A_lineal_m2) * 1e18
    
    # Errores porcentuales respecto al resultado numérico discreto
    df['Error Integral [%]'] = np.where(
        df['A_real [nm2]'] > 1e-3,
        np.abs(df['A_real [nm2]'] - df['A_GW_integral [nm2]']) / df['A_real [nm2]'] * 100.0,
        0.0
    )
    df['Error Lineal [%]'] = np.where(
        df['A_real [nm2]'] > 1e-3,
        np.abs(df['A_real [nm2]'] - df['A_GW_lineal [nm2]']) / df['A_real [nm2]'] * 100.0,
        0.0
    )
    return df
"""
#Print temporal hecho para ir comprobando totalmetne desactualizado y con el único proposito de dejarlo para reciclar, si nos viene bien, alguna represntación. Evidentemente se quitará para el final.
# Convertir a unidades legibles (nanómetros)
L_total_nm = (N_PUNTOS * DX) * 1e9  # Longitud lateral física en nm
Z_nm = Z * 1e9

# Crear la malla física en nanómetros para el render 3D
eje_nm = np.linspace(0, L_total_nm, N_PUNTOS)
X_malla, Y_malla = np.meshgrid(eje_nm, eje_nm)

# Configuración de la figura con dos paneles
fig = plt.figure(figsize=(14, 6))

# --- PANEL 1: Mapa 2D (Vista cenital / AFM) ---
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(
    Z_nm, 
    extent=[0, L_total_nm, 0, L_total_nm], 
    origin='lower', 
    cmap='viridis'
)
ax1.set_title("Topografía 2D (Mapa de alturas)", fontsize=13)
ax1.set_xlabel("x (nm)", fontsize=11)
ax1.set_ylabel("y (nm)", fontsize=11)
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Altura z (nm)", fontsize=11)

# --- PANEL 2: Superficie 3D ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Submuestreo (paso de 4) para renderizar rápido
paso = 4
surf = ax2.plot_surface(
    X_malla[::paso, ::paso], 
    Y_malla[::paso, ::paso], 
    Z_nm[::paso, ::paso],
    cmap='viridis',
    edgecolor='none',
    antialiased=True
)
ax2.set_title("Renderizado 3D de las asperezas", fontsize=13)
ax2.set_xlabel("x (nm)", fontsize=10)
ax2.set_ylabel("y (nm)", fontsize=10)
ax2.set_zlabel("z (nm)", fontsize=10)
ax2.view_init(elev=35, azim=45)
ax2.set_box_aspect((1, 1, 0.3))  # o 0.2

plt.tight_layout()
plt.savefig("superficie_topografia.png", dpi=300)
print("Gráfico generado con éxito y guardado como 'superficie_topografia.png'.")

#Print temporal del regolito 
# 1. Parámetros de prueba
RHO_OBJETIVO = 0.15          # Fracción de cobertura deseada: 15% del área cubierta
L_TOTAL = N_PUNTOS * DX       # 512 nm de lado
R_MEDIO = 20e-9              # Radio medio: 20 nm
SIGMA_R = 0.35               # Dispersión del tamaño

# 2. Llamada a las dos funciones
datos = generar_regolito_previo(RHO_OBJETIVO, L_TOTAL, r_medio=R_MEDIO, sigma_r=SIGMA_R)
Z_top, Z_bot = generar_regolito(datos, N_PUNTOS, DX, z_offset=0.0)

# 3. COMPROBACIÓN NUMÉRICA EN TERMINAL
num_granos = len(datos['x0'])
pixeles_con_polvo = np.count_nonzero(~np.isnan(Z_top))
pixeles_totales = N_PUNTOS * N_PUNTOS
rho_real = pixeles_con_polvo / pixeles_totales

cota_max_nm = np.nanmax(Z_top) * 1e9
cota_min_nm = np.nanmin(Z_bot) * 1e9
espesor_medio_nm = np.nanmean(Z_top - Z_bot) * 1e9

print("=" * 50)
print("DIAGNÓSTICO DE LA POBLACIÓN DE REGOLITO:")
print(f" - Granos generados (M): {num_granos}")
print(f" - Cobertura objetivo (theta): {RHO_OBJETIVO * 100:.1f}%")
print(f" - Cobertura real en matriz:   {rho_real * 100:.2f}%")
print(f" - Cota Z máxima (techo):      {cota_max_nm:.2f} nm")
print(f" - Cota Z mínima (suelo):      {cota_min_nm:.2f} nm")
print(f" - Espesor medio en granos:    {espesor_medio_nm:.2f} nm")
print("=" * 50)

# 4. COMPROBACIÓN VISUAL (Guardar imagen)
fig = plt.figure(figsize=(14, 6))

# Panel 1: Vista 2D (Fondo negro = vacío/NaN, islas de color = regolito)
ax1 = fig.add_subplot(1, 2, 1)
cmap_custom = plt.cm.viridis.copy()
cmap_custom.set_bad(color='black')  # Dibuja los NaN en negro sólido

im = ax1.imshow(
    Z_top * 1e9,
    extent=[0, L_TOTAL * 1e9, 0, L_TOTAL * 1e9],
    origin='lower',
    cmap=cmap_custom
)
ax1.set_title("Vista Cenital (Negro = Espacio Libre)", fontsize=12)
ax1.set_xlabel("x (nm)")
ax1.set_ylabel("y (nm)")
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Cota superior Z_top (nm)")

# Panel 2: Sección transversal 1D para comprobar el perfil de corte
# Tomamos una fila central y dibujamos el techo y el suelo de las partículas que corte
ax2 = fig.add_subplot(1, 2, 2)
fila_corte = N_PUNTOS // 2
x_eje_nm = np.arange(N_PUNTOS) * DX * 1e9

ax2.plot(x_eje_nm, Z_top[fila_corte, :] * 1e9, label="Techo (Z_top)", color="crimson", lw=2)
ax2.plot(x_eje_nm, Z_bot[fila_corte, :] * 1e9, label="Suelo (Z_bot)", color="navy", lw=2)
ax2.set_title(f"Corte transversal 1D en y = {(fila_corte * DX) * 1e9:.0f} nm", fontsize=12)
ax2.set_xlabel("x (nm)")
ax2.set_ylabel("z (nm)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("comprobacion_regolito.png", dpi=300)
print("Gráfico de control guardado como 'comprobacion_regolito.png'.")
"""


#La representación no está comentada ni revisada correctamente ya que es temporal para ver si va, ha sido hecha en gran parte por IA, para agilizar el proceso. 
# Generacion
Z_metal = generar_superficie(N_PUNTOS, DX, SIGMA_S, XI)
granos = generar_poblacion_poliedros(rho_granos=0.12, L_total=N_PUNTOS*DX, r_medio=18e-9)
Z_top, Z_bot, poliedros_3d = asentar_y_extraer_mallas(granos, Z_metal, N_PUNTOS, DX)

# Grafica 3D
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d')

# 1. Lecho metalico rugoso
X_nm, Y_nm = np.meshgrid(np.arange(N_PUNTOS)*DX*1e9, np.arange(N_PUNTOS)*DX*1e9)
paso = 3
ax.plot_surface(X_nm[::paso, ::paso], Y_nm[::paso, ::paso], Z_metal[::paso, ::paso]*1e9,
                cmap='copper', alpha=0.65, edgecolor='none')

# 2. Granos de regolito cerrados (poliedros con facetas planas reales en techo y suelo)
for caras in poliedros_3d:
    caras_nm = [c * 1e9 for c in caras]
    coleccion = Poly3DCollection(caras_nm, alpha=0.90, facecolors='#3b7da8', edgecolors='#1e4259', linewidths=0.5)
    ax.add_collection3d(coleccion)

ax.set_xlim(0, N_PUNTOS*DX*1e9)
ax.set_ylim(0, N_PUNTOS*DX*1e9)
ax.set_zlim(np.min(Z_metal)*1e9, (np.max(Z_metal) + 40e-9)*1e9)
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_zlabel("z (nm)")
ax.set_box_aspect((1, 1, 0.35))
ax.view_init(elev=22, azim=-55)

plt.tight_layout()
plt.savefig("regolito_poliedrico_cerrado.png", dpi=300)
plt.show()


#Comprobación de AGW 
# Propiedades mecánicas (ej. Aluminio-Aluminio: E=70 GPa, nu=0.33)
E_mat = 70e9
nu_mat = 0.33
E_star = 1.0 / (2.0 * (1.0 - nu_mat**2) / E_mat)

# 1. Superficie compuesta equivalente
Z1 = generar_superficie(N_PUNTOS, DX, SIGMA_S, XI)
Z2 = generar_superficie(N_PUNTOS, DX, SIGMA_S, XI)
Z_eq = Z1 + Z2

# 2. Extracción numérica discreta
alturas, radios, R_medio, rho, sigma_p = caracterizar_asperezas(Z_eq, DX)
F_num, A_num, separaciones = barrido_contacto_GW(alturas, radios, E_star, n_pasos=60)

# 3. Solución teórica analítica continua de Greenwood-Williamson (1966)
A0 = (N_PUNTOS * DX)**2
A_teorico_GW = []
F_teorico_GW = []

phi = lambda s: (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s**2)

#Printeamos los resultados con pandas en un DataFrame para luego poder añadir siempre más información como columnas con las areas analiticas y aproximadas
area_aparente_nm2 = (N_PUNTOS * DX * 1e9)**2

# Construcción del DataFrame con magnitudes físicas escaladas
df_contacto = pd.DataFrame({
    'd [nm]': separaciones * 1e9,
    'F_normal [uN]': F_num * 1e6,
    'A_real [nm2]': A_num * 1e18,
    'A_real / A_0 [%]': (A_num * 1e18 / area_aparente_nm2) * 100.0,
    'P_media [MPa]': np.where(A_num > 0, (F_num / A_num) * 1e-6, 0.0)
})

# Configuración de pandas para ver la tabla completa sin recortes en consola
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:12.4e}' if abs(x) < 1e-2 and x != 0 else f'{x:12.3f}')

print("\n" + "=" * 80)
print("RESULTADOS NUMÉRICOS DISCRETOS: BARRIDO DE CONTACTO")
print(f"Área aparente nominal A_0: {area_aparente_nm2:.2f} nm²")
print("=" * 80)
print(df_contacto)
print("=" * 80)

A_0 = (N_PUNTOS * DX)**2

# Actualizamos el DataFrame existente
df_contacto = calcular_areas_analiticas_GW(df=df_contacto,alturas_picos=alturas,R_medio=R_medio,rho=rho,sigma_p=sigma_p,A_0=A_0,E_star=E_star)

# Visualización comparativa
cols = ['d [nm]', 'F_normal [uN]', 'A_real [nm2]', 'A_GW_integral [nm2]', 'A_GW_lineal [nm2]','Error Integral [%]', 'Error Lineal [%]']
print(df_contacto[cols].to_string())
