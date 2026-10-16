# Datos

## Fuente

- **Conjunto**: *Default of Credit Card Clients* (conjunto público).
- **Repositorio**: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Descarga directa**: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
- **DOI**: 10.24432/C55S3H
- **Licencia**: Creative Commons Attribution 4.0 International (CC BY 4.0). No requiere autorización de uso; sí citar la fuente.
- **Autor / cita**: Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- **Artículo asociado**: Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications, 36*(2), 2473–2480.
- **Población**: clientes de tarjeta de crédito de un banco de Taiwán.
- **Periodo**: estados de cuenta y pagos de abril a septiembre de 2005; el objetivo es el incumplimiento del mes siguiente (octubre de 2005).
- **Moneda**: dólares taiwaneses (NT$).
- **Fecha de descarga**: [COMPLETAR]

## Archivo local

`data/raw/DATA.csv` — 30.000 filas, 25 columnas, separador `;`. Mismo contenido que el archivo original de UCI; solo se renombraron las columnas al español. **No se versiona en el repositorio** (descargar del enlace anterior).

| Checksum | Valor |
|---|---|
| MD5 | `03298a75400aa0e886c059347678d6bd` |
| SHA-256 | `b78a83dc96cb02c5b243c874cbe9b8e630c40ba5316a8b7b450b9872919c7768` |

## Correspondencia de columnas (UCI → este proyecto)

| UCI | Este proyecto | Descripción |
|---|---|---|
| ID | ID | Identificador |
| X1 (LIMIT_BAL) | MONTO_PP | Línea de crédito otorgada (NT$), individual y familiar |
| X2 (SEX) | SEXO | 1 = hombre, 2 = mujer |
| X3 (EDUCATION) | EDUCACION | 1 = posgrado, 2 = universidad, 3 = secundaria, 4 = otros (0, 5 y 6 no documentados) |
| X4 (MARRIAGE) | ESTADO_CIVIL | 1 = casado, 2 = soltero, 3 = otros (0 no documentado) |
| X5 (AGE) | EDAD | Edad en años |
| X6–X11 (PAY_0, PAY_2–PAY_6) | PAGO_M1–PAGO_M6 | Estado de pago, sept. 2005 (M1) a abr. 2005 (M6): −1 pago puntual, 1–8 meses de atraso, 9 = nueve o más (−2 y 0 no documentados) |
| X12–X17 (BILL_AMT1–6) | MONTO_FACTU_M1–M6 | Monto facturado (NT$), sept. a abr. 2005 |
| X18–X23 (PAY_AMT1–6) | PAGO_EFECT_M1–M6 | Monto pagado (NT$), sept. a abr. 2005 |
| default payment next month | Y | 1 = incumplimiento el mes siguiente |

Variables derivadas (`src/utils.py`): `PAGO_EFECT_M_MIN` es el **menor monto pagado** en los 6 meses, no el pago mínimo exigido por el banco.
