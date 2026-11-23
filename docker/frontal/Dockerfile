# Frontal de Siguiente Partida: HTML y JavaScript sin framework, servido con nginx. Se construye
# desde la raíz del repositorio:
#   docker compose build frontal
# Imagen oficial de nginx sin root (escucha en el 8080). La configuración de Firebase no va en la
# imagen: al arrancar, nginx rellena default.conf.template con las variables de entorno.
FROM nginxinc/nginx-unprivileged:stable-alpine

COPY docker/frontal/default.conf.template /etc/nginx/templates/default.conf.template
COPY frontal/ /usr/share/nginx/html/

EXPOSE 8080