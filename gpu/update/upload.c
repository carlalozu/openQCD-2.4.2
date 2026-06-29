#include "update.h"
#include "global.h"
#include "mdflds.h"
#include "uflds.h"


void update_flds_to(void)
{
   su3_dble *ud;
   mdflds_t *mdfs;

   mdfs=mdflds();
   ud=udfld();
   #pragma omp target update to(ud[:4*VOLUME+7*(BNDRY/4)])
   #pragma omp target update to((*mdfs).mom[:4*VOLUME])
   #pragma omp target update to((*mdfs).frc[:4*VOLUME+7*(BNDRY/4)])
}

void init_data_to_device(void)
{
   su3_dble *ud;
   mdflds_t *mdfs;

   mdfs=mdflds();
   ud=udfld();
   #pragma omp target enter data map(to: ud[:4*VOLUME+7*(BNDRY/4)])
   #pragma omp target enter data map(to: (*mdfs).mom[:4*VOLUME])
   #pragma omp target enter data map(to: (*mdfs).frc[:4*VOLUME+7*(BNDRY/4)])
}

void update_flds_from(void)
{
   su3_dble *ud;
   mdflds_t *mdfs;

   mdfs=mdflds();
   ud=udfld();
   #pragma omp target update from(ud[:4*VOLUME+7*(BNDRY/4)])
   #pragma omp target update from((*mdfs).mom[:4*VOLUME])
   #pragma omp target update from((*mdfs).frc[:4*VOLUME+7*(BNDRY/4)])
}