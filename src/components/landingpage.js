import React, {Component}  from 'react' ;
import {Grid,Cell} from 'react-mdl';
class Landing extends Component{

  render(){
    return (
     <div style={{width: '100%' ,margin: 'auto' }} >
       <Grid className="landing-grid">
         <Cell col={12}>
          <img
            src="https://cdn4.iconfinder.com/data/icons/science-and-technology-1-17/65/28-512.png"
            alt="ML"
            className="ML-img"
             />
             <div className="banner-text">
              <h1>Restaurant Visitor Prediction</h1>
              <hr/>
             </div>
         </Cell>

       </Grid>
     </div>
    )
  }
}

export default Landing;
