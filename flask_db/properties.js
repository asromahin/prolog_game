-----------------
<h4>Country</h4>
      <select id="Country" onchange="parentList_OnChange(this)">
            <option></option>
            <option value="France">France</option>
            <option value="Italy">Italy</option>
            <option value="Spain">Spain</option>
      </select>
      <br />
      <h4>Region</h4>
            <select id="Region"></select> 
-----------------

var childList1Data = {
      "France": ["","Bordeaux", "Champagne"],
      "Spain": ["Andalucia","Catalonia"],
      "Italy": ["Lombardy", "Tuscany"]
};

function parentList_OnChange(objParentList) {
      var child1 = document.getElementById("Region");

      // Remove all options from both child lists
      removeOptions(child1);

    // Lookup and get the array of values for child list 1, using the parents selected value
      var child1Data = childList1Data[objParentList.options[objParentList.selectedIndex].value];
    
      // Add the options to child list 1
      if (child1Data) {
        for (var i = 0; i < child1Data.length; i++) {
            child1.options[i] = new Option(child1Data[i], child1Data[i]);
          }
     }
}