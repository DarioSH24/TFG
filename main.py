import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARÁMETROS FÍSICOS Y NUMÉRICOS DEL SISTEMA
# ==============================================================================
N_PUNTOS = 512          # Resolución de la malla (512x512)
DX = 1e-9               # Resolución espacial: 1 nm por píxel
SIGMA_S = 1.5e-9        # Rugosidad RMS de la superficie metálica: 1.5 nm
XI = 10e-9              # Longitud de correlación espacial: 10 nm
# ==============================================================================
# 2. FUNCIONES MODULARES
# ==============================================================================
def generar_superficie(N,dx,sigma_s,XI):
    #1 Generamos la maya para trabajar con meshgrid
    limite=3 * XI #Generamos el limite espacial del filtro, que es tres veces la resolución
    coords= np.arange(-limite,limite+dx,dx) #Centramos el vector en el cero 
    X, Y = np.meshgrid(coords, coords)
    #2 Generamos la matriz de ruido de fondo
    ruido= np.random.normal(loc=0.0, scale=1.0, size=(N,N))
    #3 Hacemos el Kernel de correlación normalizado
    kernel= np.exp(-(X**2+Y**2)/XI**2)
    kernel_n=kernel/np.sum(kernel)#La normalización nos asegura que la suma de los pesos sea 1 y evitar alterar la señal original 
    #4 Hacemos la convolución para hacer que los puntos proximos de la matriz se afecten entre ellos
    superficie= fftconvolve(ruido,kernel_n,mode="same")
    #5 Normalizamos para tener una matriz con significado
    superficie_n=((superficie-np.mean(superficie))/np.std(superficie))*sigma_s
    return superficie_n

def generar_regolito_previo(rho_granos, L_total, r_medio=15e-9, sigma_r=0.4):
    """
    Genera los parámetros de cada una de las M partículas de regolito lunar.
    Devuelve un diccionario de vectores 1D necesarios para los calculos y representaciones.
    En este caso la densidad de granos es el cosciente entre area ocupada y libre, L referencia al lado, r_medio es el tamaño medio del grano de regolito y sigma su desviación estandar.
    """
    #Nuemro de granos    
    M = int(rho_granos * L_total**2 / ( np.pi * (r_medio**2)))
    #Generamos las posiciónes aleatoriamente y dsitibuidads uniformemente 
    x0 = np.random.uniform(0, L_total, size=M)
    y0 = np.random.uniform(0, L_total, size=M)
    #Generamos los tamños y elipticidades de forma normal
    radios = np.random.lognormal(mean=np.log(r_medio), sigma=sigma_r, size=M)
    muestreo_ba = np.random.normal(loc=0.75, scale=0.1, size=M)
    relacion_ba = np.clip(muestreo_ba, 0.4, 0.98)
    a = radios / np.sqrt(relacion_ba)
    b = radios * np.sqrt(relacion_ba)
    #Repetimos la relación para c
    muestreo_cb = np.random.normal(loc=0.75, scale=0.1, size=M)
    relacion_cb = np.clip(muestreo_cb, 0.4, 0.95)
    c = b * relacion_cb
    #La distibución de angulos evidentemente será uniforme ya que es aleatoria
    angulos = np.random.uniform(0, np.pi, size=M)    
    #La picosidad vuelve a ser normal 
    muestreo_p = np.random.normal(loc=1.0, scale=0.15, size=M)
    picosidad = np.clip(muestreo_p, 0.7, 1.4)



    return { 'x0': x0, 'y0': y0,'a': a, 'b': b, 'c': c,'theta': angulos,'p': picosidad} #Por coherencia interna la salida de esta llamada deberá ser datos_granos

def generar_regolito(datos_granos,N,dx,z_offset=0.0): #Introducimos el z_offset con un valor de cero, ya que en principio no es necesario, sin embargo puede ser util en ciertas circunstancias.
    """
    Generamos dos matrices 2.5 D con la información que hemos obtenido antes, estas matrices guardan las alturas tanto de arriba como de abajo de los granos de regolito y las posiciones donde hay particulas dentro de la matriz. 
    Las celdas que cadecen de material quedan como NAN para evitar que en la generación de la imagen se represente un valor constante.
    """
    Z_top = np.full((N, N), np.nan)
    Z_bot = np.full((N, N), np.nan)

    #Para evitar que cambios en dx o N arruinen el codigo, tratamos los datos 
    eje_fisico = np.arange(N) * dx
    # 2. Desempaquetamos los vectores analíticos
    x0 = datos_granos['x0']
    y0 = datos_granos['y0']
    a = datos_granos['a']
    b = datos_granos['b']
    c = datos_granos['c']
    theta = datos_granos['theta']
    p = datos_granos['p']
    num_granos = len(x0) #Obtenemos cuantos granos hay, M en la función anterior
    #Iteramos para cada grano y generamos la distibución de alturas y posiciones dentro de la matriz en función de los centos de las elipses y sus parametros 
    for i in range(num_granos):
        # Parámetros geométricos del grano i
        xi_0 = x0[i]
        yi_0 = y0[i]
        ai   = a[i]
        bi   = b[i]
        ci   = c[i]
        thi  = theta[i]
        pi   = p[i]


        #Delimitamos con np.clip para asegurarnos de no salir fuera del dominio en el que trabajamos que solo es la posción maxima en la que puede estar el grano teniendo en cuenta su centro y sus parametros para ahorrar recursos 
        j_min = int(np.clip(np.floor((xi_0 - ai) / dx), 0, N - 1))
        j_max = int(np.clip(np.ceil((xi_0 + ai) / dx) + 1, 0, N))
        
        i_min = int(np.clip(np.floor((yi_0 - ai) / dx), 0, N - 1))
        i_max = int(np.clip(np.ceil((yi_0 + ai) / dx) + 1, 0, N))
        if j_min >= j_max or i_min >= i_max:
            continue
        
        #Calculamos la información relevante solo en las posiciones en las que realmente puede estar el grano 
        sub_x = eje_fisico[j_min:j_max]
        sub_y = eje_fisico[i_min:i_max]
        X_loc, Y_loc = np.meshgrid(sub_x, sub_y)
        
        # Redefinimos las posiciones respecto a un nuevo centro para trabajar más comodamente
        dx_loc = X_loc - xi_0
        dy_loc = Y_loc - yi_0
        
        # Rotación por el ángulo de orientación theta
        cos_th = np.cos(thi)
        sin_th = np.sin(thi)
        x_rot =  dx_loc * cos_th + dy_loc * sin_th
        y_rot = -dx_loc * sin_th + dy_loc * cos_th
        
        # Distancia radial normalizada en el elipsoide
        rho2 = (x_rot / ai)**2 + (y_rot / bi)**2
        
        # Máscara booleana: píxeles que caen dentro del contorno elíptico
        mascara = rho2 <= 1.0
        
        if not np.any(mascara):
            continue # Si el grano es mas pequeño que un pixel se ignora 

        rho_interior = np.sqrt(rho2[mascara])
        delta_z = ci * (1.0 - rho_interior**pi)
        
        # Cotas absolutas localizadas
        z_grano_top = z_offset + delta_z
        z_grano_bot = z_offset - delta_z

        # Extraemos las coordendas de la matriz que son relevantes
        sub_top = Z_top[i_min:i_max, j_min:j_max]
        sub_bot = Z_bot[i_min:i_max, j_min:j_max]
        
        # np.fmax / np.fmin respetan los NaN previos y escogen la envolvente extrema
        sub_top[mascara] = np.fmax(sub_top[mascara], z_grano_top)
        sub_bot[mascara] = np.fmin(sub_bot[mascara], z_grano_bot)
        
        # Reinyectamos en la matriz global
        Z_top[i_min:i_max, j_min:j_max] = sub_top
        Z_bot[i_min:i_max, j_min:j_max] = sub_bot


    return Z_top, Z_bot





Z = generar_superficie(N_PUNTOS, DX, SIGMA_S, XI)


"""
#Print temporal hecho por GEMINI para comprobar que pasa, no debe estar como algo final
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
"""
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