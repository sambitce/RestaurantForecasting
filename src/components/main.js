import React from 'react' ;
import {Switch,Route} from 'react-router-dom'
import LandingPage from './landingpage' ;
import Forecast from './forecast'
const Main=  ()=> (

<Switch>
  <Route exact path = "/" component={LandingPage} />
  <Route  path = "/forecast" component={Forecast} />
</Switch>
)

export default Main;
