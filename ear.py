import  json
from    main    import *
from    utils   import u_getPath, u_whichOS

############################### MAIN ###########################################
def main():
    
    funcdict    = { 'initdb'        : initdb,
                   'earfeatures'    : earfeatures
                   }
  
    confs       = json.load(open(u_getPath('./main_conf.json')))
    confs['general']['path_op']     = u_whichOS()

    #...........................................................................
    for func in confs['functions']:
        print('_______________________________________')
        print('Function: ' + func)
        funcdict[func]( general     = confs['general'],
                        individual  = confs[func])
        print('_______________________________________')


################################################################################
if __name__ == '__main__':
    main()
